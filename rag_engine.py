# -*- coding: utf-8 -*-
"""
rag_engine.py — 文档分块、通义 Embedding、Chroma 本地持久化、混合检索

闭环：
1) 读 ``data/<学习资料 | 数据开发面经>/`` 下可索引文件 → 切块 → Embedding → Chroma；
2) 支持 **增量同步**（manifest 校验内容哈希，变更/新增才重嵌入）；
3) 单次 **上传入库**（Streamlit）保存到指定子目录后立即索引。
4) **混合检索**：BM25 关键词检索 + 向量语义检索 + RRF 重排序；
5) **动态 Top-K**：根据查询复杂度自动调整返回条数；
6) **多轮对话增强**：利用对话历史扩展检索上下文。

支持格式（扩展名）：.md .txt .markdown .pdf .docx
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, cast

import chromadb
from chromadb.api.models.Collection import Collection
from openai import APITimeoutError, OpenAI

import config

logger = logging.getLogger(__name__)

RAG_COLLECTION_NAME: str = "de_interview_kb"
# DashScope 兼容 Embedding：每批 input 条数不得超过 10，否则 400 InvalidParameter。
EMBED_BATCH_SIZE: int = 10

# 允许的扩展名（小写）；与 iter_indexable_files 一致
INDEXABLE_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".markdown", ".txt", ".pdf", ".docx"}
)

# 第一层目录映射到可读分类标签（存入 metadata kb_category）
def _infer_kb_category(rel_posix: str) -> str:
    first = (rel_posix.split("/", 1)[0] if rel_posix else "").strip()
    if first == config.KB_DIR_LEARNING or first == "学习资料":
        return "学习资料"
    if first in (config.KB_DIR_INTERVIEW, "数据开发面经", "开发面经"):
        return "面试面经"
    return first or "未分类"


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
def project_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def get_chroma_persist_path() -> Path:
    p = config.CHROMA_PERSIST_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def manifest_json_path() -> Path:
    return get_chroma_persist_path() / "kb_file_manifest.json"


def _load_manifest() -> dict[str, Any]:
    p = manifest_json_path()
    if not p.is_file():
        return {"version": 1, "files": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": 1, "files": {}}
        data.setdefault("version", 1)
        data.setdefault("files", {})
        if not isinstance(data["files"], dict):
            data["files"] = {}
        return data
    except OSError:
        return {"version": 1, "files": {}}


def _save_manifest(manifest: dict[str, Any]) -> None:
    p = manifest_json_path()
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_filename(name: str) -> str:
    base = Path(name).name.strip() or "document"
    base = re.sub(r'[<>:"/\\|?*]', "_", base)
    return base[:180] if len(base) > 180 else base


def ensure_kb_subdirs(data_root: Path | None = None) -> None:
    """确保 data/学习资料与 data/数据开发面经 存在。"""
    root = (data_root or project_data_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    interview = root / getattr(config, "KB_DIR_INTERVIEW", "数据开发面经")
    learning = root / getattr(config, "KB_DIR_LEARNING", "学习资料")
    interview.mkdir(parents=True, exist_ok=True)
    learning.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 分块
# ---------------------------------------------------------------------------
def simple_chunk_text(
    text: str,
    chunk_size: int = config.RAG_CHUNK_SIZE,
    overlap: int = config.RAG_CHUNK_OVERLAP,
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 应在 [0, chunk_size) 内")

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# 读文件为纯文本（与 tools.extract_plain_from_upload 行为尽量接近）
# ---------------------------------------------------------------------------
def _read_pdf_text(raw_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        logger.warning("未安装 pdfplumber")
        return "[PDF] 无法解析。"
    buf = io.BytesIO(raw_bytes)
    try:
        parts: list[str] = []
        with pdfplumber.open(buf) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts).strip() or "[PDF] 未提取到文本。"
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF 解析异常: %s", exc)
        return f"[PDF] 解析失败: {exc}"


def _read_docx_bytes(raw_bytes: bytes) -> str:
    try:
        from docx import Document  # type: ignore[import-untyped]
    except ImportError:
        return "[DOCX] 需要安装 python-docx。"
    try:
        doc = Document(io.BytesIO(raw_bytes))
        lines = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        body = "\n".join(lines).strip()
        return body if body else "[DOCX] 未提取到正文段落。"
    except Exception as exc:  # noqa: BLE001
        logger.warning("DOCX 解析异常: %s", exc)
        return f"[DOCX] 解析失败: {exc}"


def read_document_as_text(path: Path) -> str:
    """从磁盘读取并返回 UTF-8 纯文本占位串（失败时有提示前缀）。"""
    lower = path.name.lower()
    try:
        if lower.endswith((".md", ".markdown", ".txt")):
            return path.read_text(encoding="utf-8")
        raw = path.read_bytes()
        if lower.endswith(".pdf"):
            return _read_pdf_text(raw)
        if lower.endswith(".docx"):
            return _read_docx_bytes(raw)
        return raw.decode("utf-8", errors="replace")
    except OSError as exc:
        logger.warning("读取失败: %s (%s)", path, exc)
        return ""


def iter_indexable_files(data_dir: Path) -> list[Path]:
    if not data_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(data_dir.glob("**/*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in INDEXABLE_EXTENSIONS:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def _embed_client() -> OpenAI:
    if not config.DASHSCOPE_API_KEY:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，无法向量化。请检查 .env")
    return OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.QWEN_BASE_URL,
    )


def embed_texts(
    texts: list[str],
    *,
    on_batches_done: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """
    on_batches_done(done_count, total_count)：每完成一批 embedding 调用一次，
    ``done_count`` 为已送入模型的文本条数上界（含本批）。

    DashScope Embedding 易出现网络层 ``ConnectTimeout``，使用独立超时 ``EMBEDDING_HTTP_TIMEOUT_SECONDS``
    并在单批上做有限次退避重试。
    """
    if not texts:
        return []
    client = _embed_client()
    out: list[list[float]] = []
    model = config.QWEN_EMBEDDING_MODEL
    timeout_s = config.EMBEDDING_HTTP_TIMEOUT_SECONDS
    max_retries = max(1, config.EMBEDDING_MAX_RETRIES)
    n = len(texts)
    for i in range(0, n, EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(
                    model=model,
                    input=batch,
                    timeout=timeout_s,
                )
                for item in resp.data:
                    out.append(list(item.embedding))
                last_err = None
                break
            except APITimeoutError as exc:
                last_err = exc
                logger.warning(
                    "Embedding 批次超时 (%s/%s)，%.1fs 后重试: %s",
                    attempt + 1,
                    max_retries,
                    timeout_s,
                    exc,
                )
                time.sleep(min(8.0, 1.5 * (2**attempt)))
        if last_err is not None:
            raise last_err
        if on_batches_done is not None:
            done = min(n, i + len(batch))
            on_batches_done(done, n)
    return out


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------
def _persistent_chroma() -> Any:
    return chromadb.PersistentClient(path=str(get_chroma_persist_path()))


def get_collection(readonly: bool = True) -> Collection:
    _ = readonly
    client = _persistent_chroma()
    return client.get_or_create_collection(
        name=RAG_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def collection_chunk_count() -> int:
    try:
        return get_collection().count()
    except Exception as exc:  # noqa: BLE001
        logger.debug("读取 Chroma 条数失败: %s", exc)
        return 0


def is_vector_store_empty() -> bool:
    return collection_chunk_count() == 0


def delete_chunks_for_source(rel_posix: str) -> int:
    """删除该相对路径对应的全部向量块。"""
    col = get_collection(readonly=False)
    rows = col.get(where={"source": {"$eq": rel_posix}})
    ids = rows.get("ids") or []
    if ids:
        col.delete(ids=ids)
    return len(ids)


def _stable_chunk_id(rel_path: str, chunk_index: int, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{rel_path}::c{chunk_index}::{h}"


def chunk_document_to_payloads(
    rel_posix: str,
    *,
    kb_category: str,
    plain_text: str,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    parts = simple_chunk_text(plain_text)
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_meta: list[dict[str, Any]] = []
    for i, text in enumerate(parts):
        if not (text and text.strip()):
            continue
        cid = _stable_chunk_id(rel_posix, i, text)
        all_ids.append(cid)
        all_docs.append(text)
        all_meta.append(
            {
                "source": rel_posix,
                "chunk_index": i,
                "kb_category": kb_category,
            }
        )
    return all_ids, all_docs, all_meta


def upsert_vectors_to_collection(
    collection: Collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    vectors: list[list[float]],
) -> None:
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=cast(Any, metadatas),
        embeddings=cast(Any, vectors),
    )


# ---------------------------------------------------------------------------
# 索引进口
# ---------------------------------------------------------------------------
def index_markdown_directory(
    data_dir: Path | None = None,
    *,
    clear_before: bool = True,
) -> dict[str, Any]:
    """兼容旧名：等价于全量重建（多格式）。"""
    return index_knowledge_base_full_rebuild(data_dir=data_dir, clear_before=clear_before)


def index_knowledge_base_full_rebuild(
    data_dir: Path | None = None,
    *,
    clear_before: bool = True,
) -> dict[str, Any]:
    """
    全量重建：默认清空集合后，将 ``data/`` 下所有可索引文件灌入 Chroma，并重写 manifest。
    """
    if not config.is_api_configured():
        return {"ok": False, "error": "DASHSCOPE_API_KEY 未正确配置，无法建索引"}

    root = (data_dir or project_data_dir()).resolve()
    ensure_kb_subdirs(root)
    if not root.is_dir():
        return {"ok": False, "error": f"目录不存在: {root}"}

    client = _persistent_chroma()
    if clear_before:
        try:
            client.delete_collection(RAG_COLLECTION_NAME)
            logger.info("已删除旧集合: %s", RAG_COLLECTION_NAME)
        except Exception:  # noqa: BLE001
            logger.debug("无旧集合可删，将直接创建。")

    collection = get_collection(readonly=False)
    files = iter_indexable_files(root)
    if not files:
        _save_manifest({"version": 1, "files": {}})
        return {
            "ok": True,
            "message": f"在 {root} 下未找到可索引文件（{', '.join(sorted(INDEXABLE_EXTENSIONS))}）",
            "files": 0,
            "chunks": 0,
        }

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metas: list[dict[str, Any]] = []
    file_manifest_snippet: dict[str, Any] = {}

    for fp in files:
        rel = fp.relative_to(root).as_posix()
        kb_cat = _infer_kb_category(rel)
        body = read_document_as_text(fp)
        ids, docs, meta = chunk_document_to_payloads(rel, kb_category=kb_cat, plain_text=body)
        if not ids:
            logger.warning("无有效切块，跳过: %s", rel)
            continue
        all_ids.extend(ids)
        all_docs.extend(docs)
        all_metas.extend(meta)
        file_manifest_snippet[rel] = {"sha256": _file_sha256(fp)}

    if not all_ids:
        _save_manifest({"version": 1, "files": file_manifest_snippet})
        return {
            "ok": True,
            "message": "没有有效文本块（可能为空文件或解析失败）",
            "files": len(files),
            "chunks": 0,
        }

    vectors = embed_texts(all_docs)
    if len(vectors) != len(all_ids):
        return {"ok": False, "error": "向量条数与文本块条数不一致。"}

    collection.add(
        ids=all_ids,
        documents=all_docs,
        metadatas=cast(Any, all_metas),
        embeddings=cast(Any, vectors),
    )

    manifest: dict[str, Any] = {"version": 1, "files": file_manifest_snippet}
    _save_manifest(manifest)
    return {
        "ok": True,
        "mode": "full_rebuild",
        "data_dir": str(root),
        "files_indexed": len(files),
        "chunks": len(all_ids),
        "collection_count": collection.count(),
    }


def incremental_sync_from_disk(
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """
    增量同步：扫描 ``data/`` 下所有可索引文件，与 manifest 比对 SHA256；
    变更/新增则先删该 source 再嵌入；磁盘上已删除的文件从向量库与 manifest 移除。
    """
    if not config.is_api_configured():
        return {"ok": False, "error": "DASHSCOPE_API_KEY 未正确配置，无法建索引"}

    root = (data_dir or project_data_dir()).resolve()
    ensure_kb_subdirs(root)
    if not root.is_dir():
        return {"ok": False, "error": f"目录不存在: {root}"}

    manifest = _load_manifest()
    files_map: dict[str, Any] = manifest.setdefault("files", {})

    on_disk = {p.relative_to(root).as_posix(): p for p in iter_indexable_files(root)}

    collection = get_collection(readonly=False)
    removed = 0
    updated = 0
    skipped = 0

    # 1) 磁盘已删除：清向量 + manifest
    for old_rel in list(files_map.keys()):
        if old_rel not in on_disk:
            removed += delete_chunks_for_source(old_rel)
            files_map.pop(old_rel, None)

    # 2) 新增或内容变化
    for rel, fp in sorted(on_disk.items()):
        sha = _file_sha256(fp)
        prev = files_map.get(rel)
        if isinstance(prev, dict) and prev.get("sha256") == sha:
            skipped += 1
            continue

        delete_chunks_for_source(rel)
        kb_cat = _infer_kb_category(rel)
        body = read_document_as_text(fp)
        ids, docs, metas = chunk_document_to_payloads(rel, kb_category=kb_cat, plain_text=body)
        if not ids:
            files_map[rel] = {"sha256": sha, "chunks": 0, "note": "empty_or_unparsed"}
            updated += 1
            continue

        vectors = embed_texts(docs)
        if len(vectors) != len(ids):
            return {"ok": False, "error": f"向量化条数异常: {rel}"}
        upsert_vectors_to_collection(collection, ids, docs, metas, vectors)
        files_map[rel] = {"sha256": sha, "chunks": len(ids)}
        updated += 1

    _save_manifest(manifest)
    return {
        "ok": True,
        "mode": "incremental",
        "data_dir": str(root),
        "files_on_disk": len(on_disk),
        "files_updated_or_new": updated,
        "files_unchanged": skipped,
        "sources_removed_from_disk": removed,
        "collection_count": collection.count(),
    }


def save_upload_to_kb_folder(
    raw_bytes: bytes,
    filename: str,
    *,
    category: str,
    data_dir: Path | None = None,
) -> Path:
    """
    将上传文件保存到 ``data/学习资料/`` 或 ``data/数据开发面经/`` 下并返回绝对路径。

    category: ``"learning"`` | ``"interview"``
    """
    root = (data_dir or project_data_dir()).resolve()
    ensure_kb_subdirs(root)
    if category == "learning":
        sub = root / getattr(config, "KB_DIR_LEARNING", "学习资料")
    else:
        sub = root / getattr(config, "KB_DIR_INTERVIEW", "数据开发面经")
    sub.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_filename(filename)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = sub / f"upload_{stamp}_{safe}"
    dest.write_bytes(raw_bytes)
    return dest.resolve()


def ingest_uploaded_bytes(
    raw_bytes: bytes,
    filename: str,
    *,
    category_key: str,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """
    将文件保存到 ``data/<面经或学习资料>/`` 后，对 **整个 data/** 执行一次增量同步：
    manifest 比对 SHA256，新增/变更则重嵌入，磁盘已删文件则从向量库移除。

    （与早前实现一致：**上传后跑一次** ``incremental_sync_from_disk``。）
    """
    if not config.is_api_configured():
        return {"ok": False, "error": "DASHSCOPE_API_KEY 未正确配置，无法建索引"}

    path = save_upload_to_kb_folder(
        raw_bytes, filename, category=category_key, data_dir=data_dir
    )
    root = (data_dir or project_data_dir()).resolve()
    rel = path.relative_to(root).as_posix()
    sync = incremental_sync_from_disk(root)
    if not sync.get("ok"):
        return sync

    files_map = _load_manifest().get("files") or {}
    entry = files_map.get(rel, {}) if isinstance(files_map.get(rel), dict) else {}

    return {
        "ok": True,
        "saved_path": str(path),
        "relative": rel,
        "chunks": entry.get("chunks"),
        "sync_detail": sync,
    }
# ---------------------------------------------------------------------------
# BM25 关键词检索：用 jieba 分词 + rank_bm25 实现本地倒排索引
# ---------------------------------------------------------------------------
# jieba 延迟导入，避免首次 import rag_engine 就加载分词词典（耗时约 1-2s）
_jieba_imported = False
_jieba_lcut = None


def _ensure_jieba() -> Any:
    """延迟加载 jieba 分词函数，返回 jieba.lcut。"""
    global _jieba_imported, _jieba_lcut  # noqa: PLW0603
    if not _jieba_imported:
        try:
            import jieba as _jb
            _jieba_lcut = _jb.lcut
            _jieba_imported = True
        except ImportError:
            logger.warning("未安装 jieba，BM25 检索将使用简单字符切分（效果较差）")
            # 降级：按字符拆分，至少能跑通
            _jieba_lcut = lambda text: list(text)  # type: ignore[assignment]
            _jieba_imported = True
    return _jieba_lcut


def tokenize_chinese(text: str) -> list[str]:
    """
    对中文文本进行分词，过滤掉单字符标点和空格。

    为什么需要分词：
    BM25 是基于「词项」的统计检索模型，直接按字切分效果差（如"数据仓库"被拆成4个字）；
    jieba 能识别中文词汇，让"数据仓库"作为一个整体被索引，检索更准。
    """
    lcut = _ensure_jieba()
    tokens = lcut(text or "")
    # 过滤空白和单字符标点（如"、""，""。"），保留有意义的词
    return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]


def bm25_search(
    query: str,
    all_documents: list[str],
    all_ids: list[str],
    top_k: int = 10,
) -> list[tuple[str, str, float]]:
    """
    用 BM25 算法对本地文档做关键词检索。

    参数：
        query: 用户查询文本
        all_documents: 向量库中所有文档的纯文本列表
        all_ids: 与 all_documents 一一对应的 chunk ID 列表
        top_k: 返回前 K 个结果

    返回：[(chunk_id, document_text, bm25_score), ...] 按分数从高到低排列。

    原理简述：
    BM25（Best Matching 25）是信息检索中的经典算法。它根据查询中每个词在文档中的
    出现频率（TF）和在整个文档集合中的稀有程度（IDF）来给文档打分。
    IDF 越高的词（如"Flink"比"的"更稀有），匹配到时给文档加的分越多。
    """
    if not query.strip() or not all_documents:
        return []

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("未安装 rank_bm25，跳过 BM25 检索")
        return []

    # 对每篇文档做分词
    tokenized_docs = [tokenize_chinese(doc) for doc in all_documents]
    bm25 = BM25Okapi(tokenized_docs)

    # 对查询做分词
    tokenized_query = tokenize_chinese(query)
    if not tokenized_query:
        return []

    # BM25 打分
    scores = bm25.get_scores(tokenized_query)

    # 按分数从高到低排序，取 top_k（只返回分数 > 0 的结果）
    scored_results: list[tuple[str, str, float]] = []
    for i, score in enumerate(scores):
        if score > 0:
            scored_results.append((all_ids[i], all_documents[i], float(score)))
    scored_results.sort(key=lambda x: x[2], reverse=True)
    return scored_results[:top_k]


# ---------------------------------------------------------------------------
# 向量检索：带相似度分数过滤的增强版
# ---------------------------------------------------------------------------
def vector_search_with_scores(
    query: str,
    top_k: int = 10,
    similarity_threshold: float | None = None,
) -> list[tuple[str, str, float]]:
    """
    向量语义检索，返回 (chunk_id, document_text, similarity_score) 列表。

    与原来的 rag_search 不同：
    1) 同时返回 Chroma 的距离分数（余弦距离）；
    2) 根据 similarity_threshold 过滤掉不相关的结果。

    关于分数：
    Chroma 使用余弦距离（cosine distance），值越小表示越相似（0 = 完全相同）。
    但对用户来说，我们把它转换为「相似度」= 1 - distance，值越大越相关（0~1）。
    """
    threshold = similarity_threshold if similarity_threshold is not None else config.RAG_SIMILARITY_THRESHOLD
    if not (query or "").strip():
        return []

    if is_vector_store_empty():
        return []

    if not config.is_api_configured():
        return []

    try:
        qvec = embed_texts([query.strip()])[0]
    except Exception as exc:  # noqa: BLE001
        logger.exception("向量检索：问题向量化失败: %s", exc)
        return []

    try:
        col = get_collection()
        # 同时获取 documents 和 distances，用于后续过滤和重排序
        # distances 是余弦距离（0~2），与相似度的关系：similarity = 1 - distance
        res = col.query(
            query_embeddings=[qvec],
            n_results=min(top_k, max(1, col.count())),
            include=["documents", "distances", "metadatas"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chroma 查询失败: %s", exc)
        return []

    docs = res.get("documents") or [[]]
    ids = res.get("ids") or [[]]
    distances = res.get("distances") or [[]]
    docs_list = docs[0] if docs else []
    ids_list = ids[0] if ids else []
    dist_list = distances[0] if distances else []

    results: list[tuple[str, str, float]] = []
    for i, doc in enumerate(docs_list):
        if not isinstance(doc, str) or not doc.strip():
            continue
        dist = float(dist_list[i]) if i < len(dist_list) else 1.0
        # 余弦距离转相似度：distance 越小越相似
        similarity = max(0.0, 1.0 - dist / 2.0)
        chunk_id = str(ids_list[i]) if i < len(ids_list) else ""
        # 过滤低于阈值的结果
        if similarity >= threshold:
            results.append((chunk_id, doc.strip(), similarity))
    return results


# ---------------------------------------------------------------------------
# RRF 混合重排序（Reciprocal Rank Fusion）
# ---------------------------------------------------------------------------
def rrf_fusion(
    bm25_results: list[tuple[str, str, float]],
    vector_results: list[tuple[str, str, float]],
    *,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
    k: int = 60,
) -> list[tuple[str, str, float, str]]:
    """
    Reciprocal Rank Fusion (RRF)：将两种检索结果融合为统一排序。

    为什么要融合：
    - BM25 擅长精确关键词匹配（如搜"Flink checkpoint"能精准命中含这个词的段落）；
    - 向量检索擅长语义理解（如搜"实时计算容错"能匹配到讲 Flink checkpoint 的段落）。
    两者互补，融合后的结果通常比任一单一方法都更好。

    RRF 算法原理：
    对于第 i 个结果，其 RRF 分数 = weight / (k + rank_i)，其中 rank_i 是该结果在
    原始排序中的位置（从 1 开始）。k 是一个常数（默认 60），用于平滑排名靠前的
    结果之间的分数差距。最终分数是两种检索的加权和。

    返回：[(chunk_id, document_text, rrf_score, source_hint), ...]
        source_hint 标记该结果来自 "bm25+vector"（两者都有）、"vector" 或 "bm25"。
    """
    # 给每个结果建一个 {chunk_id -> (document, rank)} 的映射
    bm25_rank_map: dict[str, tuple[str, int]] = {}
    for rank, (cid, doc, _score) in enumerate(bm25_results, 1):
        bm25_rank_map[cid] = (doc, rank)

    vector_rank_map: dict[str, tuple[str, int]] = {}
    for rank, (cid, doc, _score) in enumerate(vector_results, 1):
        vector_rank_map[cid] = (doc, rank)

    # 收集所有出现过的 chunk_id
    all_chunk_ids = set(bm25_rank_map.keys()) | set(vector_rank_map.keys())

    fused: list[tuple[str, str, float, str]] = []
    for cid in all_chunk_ids:
        score = 0.0
        doc_text = ""
        in_bm25 = cid in bm25_rank_map
        in_vector = cid in vector_rank_map

        if in_bm25:
            doc_text = bm25_rank_map[cid][0]
            rank = bm25_rank_map[cid][1]
            score += bm25_weight / (k + rank)

        if in_vector:
            doc_text = doc_text or vector_rank_map[cid][0]
            rank = vector_rank_map[cid][1]
            score += vector_weight / (k + rank)

        # 标记来源：帮助调试和理解
        if in_bm25 and in_vector:
            source = "bm25+vector"
        elif in_vector:
            source = "vector"
        else:
            source = "bm25"

        fused.append((cid, doc_text, score, source))

    # 按 RRF 分数从高到低排序
    fused.sort(key=lambda x: x[2], reverse=True)
    return fused


# ---------------------------------------------------------------------------
# 动态 Top-K：根据查询复杂度自动调整返回条数
# ---------------------------------------------------------------------------
def compute_dynamic_top_k(query: str, base_k: int | None = None) -> int:
    """
    根据用户查询的复杂度，动态决定返回多少条检索结果。

    为什么要动态调整：
    - 简单问题（如"什么是数仓"）只需 3-4 条结果就够了，太多反而引入噪音；
    - 复杂问题（如"对比 Flink 和 Spark 的 exactly-once 实现"）需要更多结果来
      全面回答，可能需要 12-15 条。

    判断策略：
    1) 查询长度：越长通常越复杂
    2) 关键词数量：技术术语越多，问题越专业
    3) 是否包含对比/分析类词汇：这类问题需要更多信息
    """
    base = base_k if base_k is not None else config.RAG_TOP_K
    min_k = config.RAG_TOP_K_MIN
    max_k = config.RAG_TOP_K_MAX

    q = (query or "").strip()
    if not q:
        return base

    # 因素 1：查询长度（字符数）
    length_score = min(len(q) / 50.0, 1.0)  # 50 字以上满分

    # 因素 2：技术关键词密度
    tech_keywords = [
        "spark", "flink", "hive", "hadoop", "kafka", "sql", "python",
        "scala", "数仓", "数据仓库", "etl", "实时", "离线", "数据湖",
        "checkpoint", "shuffle", "join", "group by", "窗口", "倾斜",
    ]
    q_lower = q.lower()
    keyword_hits = sum(1 for kw in tech_keywords if kw in q_lower)
    keyword_score = min(keyword_hits / 3.0, 1.0)  # 命中 3 个以上关键词满分

    # 因素 3：对比/分析类词汇（这些问题需要更多信息）
    analysis_words = ["对比", "比较", "区别", "差异", "优缺点", "分析", "为什么",
                      "如何", "怎么", "原理", "流程", "设计", "方案", "权衡"]
    analysis_hits = sum(1 for w in analysis_words if w in q)
    analysis_score = min(analysis_hits / 2.0, 1.0)

    # 综合打分（加权平均）
    combined = 0.4 * length_score + 0.3 * keyword_score + 0.3 * analysis_score

    # 映射到 [min_k, max_k] 范围
    dynamic_k = int(min_k + combined * (max_k - min_k))
    return max(min_k, min(max_k, dynamic_k))


# ---------------------------------------------------------------------------
# 多轮对话检索增强：利用对话历史扩展检索上下文
# ---------------------------------------------------------------------------
def build_multi_turn_search_context(
    current_query: str,
    session_history: list[dict[str, str]] | None = None,
    max_history_turns: int = 3,
    max_total_length: int = 2000,
) -> str:
    """
    把多轮对话历史与当前查询拼接，生成更好的检索输入。

    为什么要增强：
    多轮对话中，用户常说"那上面那个呢""再细一点""Flink 那个怎么实现的"——
    这些指代性问题如果不结合上下文，单句检索会搜偏。把前面 2-3 轮的关键信息
    拼进来，能让检索 query 更完整。

    策略：
    1) 取最近 max_history_turns 轮的用户消息（不含系统/助手）
    2) 提取其中的关键信息片段
    3) 与当前查询拼接，总长度不超过 max_total_length

    参数：
        current_query: 当前用户输入
        session_history: 完整对话历史 [{"role": "user"/"assistant", "content": ...}, ...]
        max_history_turns: 取最近几轮历史
        max_total_length: 拼接后的最大字符数
    """
    cur = (current_query or "").strip()
    if not cur:
        return cur
    if not session_history:
        return cur

    # 只取用户角色的消息作为历史上下文（助手的回复通常不是检索目标）
    user_messages: list[str] = []
    for msg in reversed(session_history):
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if content and content != cur:
                user_messages.append(content)
        if len(user_messages) >= max_history_turns:
            break

    if not user_messages:
        return cur

    # 倒序还原时间线（从旧到新）
    user_messages.reverse()

    # 拼接：历史消息 + 当前查询
    history_part = " | ".join(m[:300] for m in user_messages)
    combined = f"{history_part}\n---\n{cur}"
    return combined[:max_total_length]


# ---------------------------------------------------------------------------
# 从向量库全量加载文档（用于 BM25 索引构建）
# ---------------------------------------------------------------------------
def _load_all_documents_for_bm25() -> tuple[list[str], list[str]]:
    """
    从 Chroma 集合中加载所有文档和 ID，用于构建 BM25 索引。

    为什么需要单独加载：
    BM25 是「内存中」的算法，需要一次性拿到所有文档来建倒排索引。
    与向量检索不同（向量检索在 Chroma 内部完成），BM25 的分词和打分都在 Python 侧。

    返回：(id_list, document_list)，一一对应。
    """
    try:
        col = get_collection()
        count = col.count()
        if count == 0:
            return [], []
        # 一次性获取所有文档（Chroma 支持 get 方法不传 ids 时返回全部）
        result = col.get(include=["documents"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        return ids, docs
    except Exception as exc:  # noqa: BLE001
        logger.exception("加载全部文档用于 BM25 失败: %s", exc)
        return [], []


def _log_retrieval_hit_files(chunk_ids: list[str]) -> None:
    """
    根据 Chroma 中块的 ``metadata.source``（相对路径）打日志，便于确认「命中了哪些原文文件」。
    """
    if not chunk_ids:
        return
    try:
        col = get_collection()
        res = col.get(ids=list(chunk_ids), include=["metadatas"])
        metas = res.get("metadatas") or []

        def _meta_src(meta: dict[str, Any]) -> str:
            v = meta.get("source")
            return str(v).strip() if v is not None else ""

        uniq_sources = sorted(
            {_meta_src(m) for m in metas if isinstance(m, dict) and _meta_src(m)}
        )
        by_cat: dict[str, list[str]] = {}
        for m in metas:
            if not isinstance(m, dict):
                continue
            src = _meta_src(m)
            if not src:
                continue
            cat = str(m.get("kb_category") or "未标注").strip()
            by_cat.setdefault(cat, []).append(src)
        for cat_key in list(by_cat.keys()):
            by_cat[cat_key] = sorted(set(by_cat[cat_key]))
        if uniq_sources:
            logger.info(
                "检索命中知识库文件（路径相对 data/，去重共 %s 个）: %s",
                len(uniq_sources),
                " | ".join(uniq_sources[:25]) + (" | …" if len(uniq_sources) > 25 else ""),
            )
        # 可按分类看一眼（若有 kb_category）
        if by_cat:
            logger.debug("命中按 kb_category 分组: %s", by_cat)
    except Exception as exc:  # noqa: BLE001
        logger.debug("解析命中文件名跳过: %s", exc)


# ---------------------------------------------------------------------------
# 主检索入口：混合检索（BM25 + 向量 + RRF 重排序）
# ---------------------------------------------------------------------------
def rag_search(
    query: str,
    top_k: int | None = None,
    *,
    use_hybrid: bool = True,
    session_history: list[dict[str, str]] | None = None,
) -> list[str]:
    """
    混合检索主入口：BM25 关键词 + 向量语义 → RRF 融合重排序。

    与旧版 rag_search 的区别：
    1) 新增 BM25 关键词检索路径；
    2) 用 RRF（Reciprocal Rank Fusion）融合两种结果；
    3) 支持动态 Top-K；
    4) 支持多轮对话历史增强。

    参数：
        query: 用户查询
        top_k: 指定返回条数（None 时自动动态计算）
        use_hybrid: 是否启用混合检索（False 时退化为纯向量检索，与旧行为一致）
        session_history: 对话历史，用于多轮检索增强

    返回：检索到的文本块列表（纯字符串），与旧接口完全兼容。
    """
    k = top_k if top_k is not None else compute_dynamic_top_k(query)
    q = (query or "").strip()
    if not q:
        return []

    if is_vector_store_empty():
        logger.warning("Chroma 中暂无向量，请先建索引或增量同步")
        return []

    # 1) 向量语义检索（核心路径，始终执行）
    try:
        vector_results = vector_search_with_scores(q, top_k=k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("向量检索异常: %s", exc)
        vector_results = []

    # 如果不启用混合检索，直接用向量结果（与旧版行为一致）
    if not use_hybrid:
        picks = vector_results[:k]
        _log_retrieval_hit_files([cid for cid, _, _ in picks])
        return [doc for _, doc, _ in picks]

    # 2) BM25 关键词检索（并行执行，失败不影响主流程）
    bm25_results: list[tuple[str, str, float]] = []
    try:
        all_ids, all_docs = _load_all_documents_for_bm25()
        if all_docs:
            # BM25 需要的检索量可以比最终返回的 k 多一些，给 RRF 融合留余量
            bm25_results = bm25_search(q, all_docs, all_ids, top_k=k * 2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("BM25 检索异常，降级为纯向量检索: %s", exc)

    # 3) 如果 BM25 没有结果，直接返回向量结果
    if not bm25_results:
        picks = vector_results[:k]
        _log_retrieval_hit_files([cid for cid, _, _ in picks])
        return [doc for _, doc, _ in picks]

    # 4) RRF 融合重排序
    alpha = config.RAG_HYBRID_ALPHA
    fused = rrf_fusion(
        bm25_results,
        vector_results,
        bm25_weight=1.0 - alpha,
        vector_weight=alpha,
    )

    # 5) 取 top_k 条，去重（以防同一文档被两种方法各命中一次）
    seen: set[str] = set()
    final_ids: list[str] = []
    final: list[str] = []
    for cid, doc, score, source in fused:
        if doc in seen:
            continue
        seen.add(doc)
        final_ids.append(cid)
        final.append(doc)
        if len(final) >= k:
            break

    _log_retrieval_hit_files(final_ids)

    logger.info(
        "混合检索完成: query=%s, bm25_hits=%d, vector_hits=%d, fused=%d, returned=%d",
        q[:50], len(bm25_results), len(vector_results), len(fused), len(final),
    )
    return final


def placeholder_rag_search(query: str, top_k: int = config.RAG_TOP_K) -> list[str]:
    """兼容旧接口。"""
    return rag_search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# 索引健康检查与监控（工程化）
# ---------------------------------------------------------------------------
def check_index_health() -> dict[str, Any]:
    """
    检查向量库与 manifest 的一致性，返回健康报告。

    检查项：
    1) Chroma 集合是否存在、有多少条向量
    2) manifest 中记录的文件数 vs 实际磁盘文件数
    3) manifest 中的 chunk 数 vs Chroma 实际 chunk 数
    4) 是否有 orphan 块（manifest 中已删除的文件在向量库中仍有残留）
    """
    report: dict[str, Any] = {
        "ok": True,
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    # 检查 1: Chroma 集合状态
    try:
        col = get_collection()
        chunk_count = col.count()
        report["checks"]["chroma_chunks"] = chunk_count
    except Exception as exc:
        report["ok"] = False
        report["errors"].append(f"Chroma 集合不可用: {exc}")
        return report

    # 检查 2: manifest 一致性
    manifest = _load_manifest()
    manifest_files = manifest.get("files", {})
    report["checks"]["manifest_files"] = len(manifest_files)

    root = project_data_dir().resolve()
    on_disk = {p.relative_to(root).as_posix() for p in iter_indexable_files(root)}
    report["checks"]["disk_files"] = len(on_disk)

    # 3) 检查 orphan：manifest 中有但磁盘上已删除的文件
    orphan_files = set(manifest_files.keys()) - on_disk
    if orphan_files:
        report["warnings"].append(
            f"manifest 中有 {len(orphan_files)} 个文件在磁盘上已删除（orphan）: "
            + ", ".join(sorted(orphan_files)[:5])
            + ("..." if len(orphan_files) > 5 else "")
        )

    # 4) 检查新增文件：磁盘上有但 manifest 中没有的
    new_files = on_disk - set(manifest_files.keys())
    if new_files:
        report["warnings"].append(
            f"磁盘上有 {len(new_files)} 个新文件未被索引: "
            + ", ".join(sorted(new_files)[:5])
            + ("..." if len(new_files) > 5 else "")
        )

    # 5) 检查 manifest 中记录的 chunk 数与实际是否一致
    manifest_total_chunks = sum(
        info.get("chunks", 0) for info in manifest_files.values() if isinstance(info, dict)
    )
    report["checks"]["manifest_total_chunks"] = manifest_total_chunks
    if manifest_total_chunks > 0 and abs(manifest_total_chunks - chunk_count) > 5:
        report["warnings"].append(
            f"manifest 记录 {manifest_total_chunks} chunks，但 Chroma 实际有 {chunk_count}，"
            "差值超过阈值，建议执行全量重建"
        )

    # 6) 检查索引元数据
    report["checks"]["embedding_model"] = getattr(config, "QWEN_EMBEDDING_MODEL", "unknown")
    report["checks"]["chunk_size"] = getattr(config, "RAG_CHUNK_SIZE", 500)
    report["checks"]["chunk_overlap"] = getattr(config, "RAG_CHUNK_OVERLAP", 50)

    return report


def get_index_stats() -> dict[str, Any]:
    """
    获取向量库的基本统计信息，供 UI 或监控使用。

    返回：包含文件数、chunk 数、各分类数量等的字典。
    """
    stats: dict[str, Any] = {}
    try:
        col = get_collection()
        stats["total_chunks"] = col.count()
    except Exception:
        stats["total_chunks"] = 0

    # 按分类统计
    try:
        col = get_collection()
        count = col.count()
        if count > 0:
            result = col.get(include=["metadatas"])
            metas = result.get("metadatas") or []
            category_counts: dict[str, int] = {}
            for meta in metas:
                if isinstance(meta, dict):
                    cat = str(meta.get("kb_category", "未分类"))
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            stats["by_category"] = category_counts
        else:
            stats["by_category"] = {}
    except Exception:
        stats["by_category"] = {}

    # manifest 信息
    manifest = _load_manifest()
    stats["manifest_files"] = len(manifest.get("files", {}))
    stats["manifest_version"] = manifest.get("version", "unknown")

    return stats


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="知识库索引 / 增量同步（DataAgent）")
    ap.add_argument(
        "cmd",
        nargs="?",
        default="incremental",
        choices=("incremental", "rebuild"),
        help="incremental=按 manifest 增量；rebuild=清空集合全量重建",
    )
    args = ap.parse_args()

    if args.cmd == "rebuild":
        print("=== 全量重建（会清空向量集合）===")
        stat = index_knowledge_base_full_rebuild(clear_before=True)
    else:
        print("=== 增量同步 data/ ===")
        stat = incremental_sync_from_disk()

    print(stat)
    ok = stat.get("ok") and stat.get("collection_count", stat.get("chunks", 1)) >= 0
    if ok and stat.get("collection_count") or stat.get("chunks"):
        print("\n=== 试检索 ===")
        for i, t in enumerate(rag_search("Spark Shuffle 在面试里怎么考察？", top_k=2), 1):
            print(f"--- 命中 {i} ---\n{t[:320]}...\n")
