# -*- coding: utf-8 -*-
"""
rag_engine.py — 文档分块、通义 Embedding、Chroma 本地持久化、向量检索

闭环：
1) 读 ``data/<学习资料 | 数据开发面经>/`` 下可索引文件 → 切块 → Embedding → Chroma；
2) 支持 **增量同步**（manifest 校验内容哈希，变更/新增才重嵌入）；
3) 单次 **上传入库**（Streamlit）保存到指定子目录后立即索引。

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
from typing import Any, cast

import chromadb
from chromadb.api.models.Collection import Collection
from openai import OpenAI

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


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = _embed_client()
    out: list[list[float]] = []
    model = config.QWEN_EMBEDDING_MODEL
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(
            model=model,
            input=batch,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        for item in resp.data:
            out.append(list(item.embedding))
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
    保存上传文件到磁盘后，对本文件做一次「增量」（必然更新 manifest 与向量）。
    """
    path = save_upload_to_kb_folder(
        raw_bytes, filename, category=category_key, data_dir=data_dir
    )
    root = (data_dir or project_data_dir()).resolve()
    rel = path.relative_to(root).as_posix()
    before = incremental_sync_from_disk(root)
    if not before.get("ok"):
        return before
    return {
        "ok": True,
        "saved_path": str(path),
        "relative": rel,
        "sync_detail": before,
        "hint": "已写入 data/ 并同步向量；也可点「增量同步」扫描整个 data/。",
    }


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------
def rag_search(query: str, top_k: int | None = None) -> list[str]:
    k = top_k if top_k is not None else config.RAG_TOP_K
    if not (query or "").strip():
        return []

    if is_vector_store_empty():
        logger.warning("Chroma 中暂无向量，请先建索引或增量同步")
        return []

    if not config.is_api_configured():
        logger.warning("无 API Key，无法把问题向量化，跳过 RAG")
        return []

    try:
        qvec = embed_texts([query.strip()])[0]
    except Exception as exc:  # noqa: BLE001
        logger.exception("问题向量化失败: %s", exc)
        return []

    try:
        col = get_collection()
        res = col.query(
            query_embeddings=[qvec],
            n_results=min(k, max(1, col.count())),
            include=["documents"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chroma 查询失败: %s", exc)
        return []

    docs = res.get("documents") or []
    if not docs or not docs[0]:
        return []
    return [d for d in docs[0] if isinstance(d, str) and d.strip()]


def placeholder_rag_search(query: str, top_k: int = config.RAG_TOP_K) -> list[str]:
    return rag_search(query, top_k=top_k)


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
