# -*- coding: utf-8 -*-
"""
tools.py — 工具层（Tool Layer）

实现需求中的：
- analyze_jd：JD 分析与关键词提取；
- analyze_resume_gap：简历与 JD 对齐度 / 技能缺口；
- rag_search_tools：对齐「工具层可调用的知识检索」门面（封装 rag_engine）；
- mock_interview_state：会话状态读取（面试官状态机在 agent_brain，此处仅常量与只读快照）。

中文注释说明「输入从哪来、输出到哪去」，便于你与 Agent Brain 连线调试。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 与聊天框拼接有关的约定（务必与 app._maybe_append_upload_text 保持一致）
# ---------------------------------------------------------------------------
_UPLOAD_MARK = "[上传文件内容]"


# ---------------------------------------------------------------------------
# analyze_jd
# ---------------------------------------------------------------------------
def analyze_jd(jd_text: str, use_llm: bool = False, llm_caller: Any = None) -> str:
    """
    对岗位描述（JD）做分析：技术栈总结、高频考点等。

    llm_caller 需实现 ``simple_chat(system_prompt: str, user_prompt: str) -> str``
    （与本项目 QwenClient 一致）。
    """
    if not (jd_text or "").strip():
        return "未检测到有效的 JD 文本，请粘贴完整岗位描述。"

    if use_llm and llm_caller is not None:
        try:
            system = (
                "你是资深数据开发（DE）方向的面试官与技术招聘顾问。请用简洁中文结构化输出：\n"
                "1）岗位核心职责；2）技术要求与数据栈；3）可能的高频面试与笔试考点；"
                "4）软性要求（若有）。不要编造 JD 中出现的技术名词。"
            )
            user = f"以下是需要分析的 JD 全文：\n\n{jd_text[:12000]}"
            return llm_caller.simple_chat(system_prompt=system, user_prompt=user)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM 分析 JD 失败: %s", exc)
            return "抱歉，分析 JD 时服务暂时不可用，已降级为简单关键词统计。"

    return _mock_analyze_jd_local(jd_text)


def _mock_analyze_jd_local(jd_text: str) -> str:
    """无 LLM 时的本地占位：关键词表匹配。"""
    tech_keywords = [
        "Spark",
        "Flink",
        "Hive",
        "Hadoop",
        "Kafka",
        "SQL",
        "Python",
        "Java",
        "Scala",
        "Airflow",
        "dbt",
        "StarRocks",
        "ClickHouse",
        "Doris",
        "数据仓库",
        "数仓",
        "ETL",
        "实时",
        "离线",
    ]
    found: list[str] = []
    lower = jd_text
    for kw in tech_keywords:
        if kw.lower() in lower.lower() or kw in jd_text:
            if kw not in found:
                found.append(kw)

    lines = [
        "【本地模拟版 JD 分析】未调用大模型，仅为关键词/要点占位。",
        f"- 在文本中匹配到的可能技术关键词：{', '.join(found) if found else '（未匹配到预置词表，可接入 LLM）'}",
        f"- 文本长度约 {len(jd_text)} 字。",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 聊天文本分割：前缀（用户话术 + 可能粘贴的 JD）与后缀（上传文件正文）
# ---------------------------------------------------------------------------
def split_user_message_and_upload_content(full_text: str) -> tuple[str, str]:
    """
    返回 (prefix, uploaded_plain)。

    - prefix：上传标记之前的全部内容（常为「请分析一下」或整段粘贴的 JD）。
    - uploaded_plain：紧跟「[上传文件内容]」后面的解析正文（通常为简历或 JD 文件）。
    """
    ft = full_text or ""
    if _UPLOAD_MARK not in ft:
        return ft.strip(), ""
    idx = ft.index(_UPLOAD_MARK)
    prefix = ft[:idx].strip()
    tail = ft[idx + len(_UPLOAD_MARK) :].strip()
    return prefix, tail


def read_resume_plain_from_session(session: dict[str, Any] | None) -> str:
    """
    侧栏最近一次上传的文件解析出的纯文本——在模拟面试／Gap 中我们约定为简历。
    （若你以后支持「JD 上传」与「简历上传」分立，再在 session 中加类型字段。）
    """
    if not isinstance(session, dict):
        return ""
    name = session.get("last_upload_name")
    raw = session.get("last_upload_bytes")
    if not name or raw is None:
        return ""
    return extract_plain_from_upload(str(name), raw).strip()


# ---------------------------------------------------------------------------
# analyze_resume_gap
# ---------------------------------------------------------------------------
def analyze_resume_gap(
    resume_plain: str,
    jd_plain: str,
    *,
    use_llm: bool = True,
    llm_caller: Any | None = None,
) -> str:
    """
    对比简历与 JD，给出技能匹配度与改进建议。

    resume_plain / jd_plain 应由上层从对话与缓存中拼装好；
    任一侧过短时返回提示性字符串，不调 LLM，避免 hallucination。
    """
    rp = (resume_plain or "").strip()
    jp = (jd_plain or "").strip()
    if len(rp) < 80:
        return (
            "简历正文过短或未上传。请先**在侧栏上传简历 PDF/MD**，"
            "并在同一段话里或上一轮对话中**粘贴完整 JD** 后再发起「对标 / Gap」分析。"
        )
    if len(jp) < 60:
        return (
            "JD 正文过短或缺失。请将**岗位描述粘贴到对话框**（可放在上传简历前的文字里），"
            "或使用「JD 分析」让系统缓存后再做 Gap。"
        )

    if use_llm and llm_caller is not None:
        try:
            system = (
                "你是数据开发方向的技术招聘顾问。请只做「简历 vs 岗位 JD」的对照分析，"
                "**不要编造简历或 JD 里没有的经历与技能**。用中文 Markdown，务必包含小节：\n"
                "## 匹配强项\n"
                "## 明显缺口（按 JD 要求逐条对齐）\n"
                "## 可补充在项目/简历侧的重点\n"
                "## 学习与面试建议（可操作）\n"
            )
            user = (
                f"【简历全文节选，不超过约 12000 字】\n{rp[:12000]}\n\n"
                f"【岗位 JD 节选，不超过约 12000 字】\n{jp[:12000]}\n"
            )
            return llm_caller.simple_chat(system_prompt=system, user_prompt=user)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gap 分析 LLM 失败: %s", exc)
            return _resume_gap_fallback_local(rp, jp)

    return _resume_gap_fallback_local(rp, jp)


def _resume_gap_fallback_local(resume_plain: str, jd_plain: str) -> str:
    """无 LLM 时：用词表粗算 JD 中出现的技能词是否在简历里也出现。"""
    tech_candidates = (
        "spark flink hive hadoop kafka sql python java scala airflow dbt starrocks "
        "clickhouse doris 数仓 数据仓库 etl 离线 实时 数据湖 flinkcdc"
    ).split()
    jd_l = jd_plain.lower()
    cv_l = resume_plain.lower()
    hit_jd = [w for w in tech_candidates if w in jd_l]
    only_jd = [w for w in hit_jd if w not in cv_l]
    return (
        "【本地模拟 Gap】未调用大模型，仅从关键词层面粗比（仅供参考）。\n"
        f"- JD 中出现的部分技术词是否在简历中也有："
        f"共 {len(hit_jd)} 项在 JD 侧命中。\n"
        f"- JD 中提到但简历中未直观出现的词示例（至多 15 项）："
        f"{', '.join(only_jd[:15]) or '（无或未命中预设词表）'}\n"
        "- 建议使用 LLM 模式获得完整段落级分析。\n"
    )


def sync_gap_sticky_caches(session: dict[str, Any], user_full_text: str) -> None:
    """
    在每轮开始时更新「简历/JD」粘滞缓冲，便于分轮对话仍能拼出 Gap。
    """
    rp = read_resume_plain_from_session(session)
    if len(rp.strip()) >= 80:
        session["resume_sticky_plain"] = rp.strip()[:20000]

    pfx, suf = split_user_message_and_upload_content(user_full_text)
    fname = str(session.get("last_upload_name") or "").lower()
    cand_jd = ""
    if suf.strip() and any(k in fname for k in ("jd", "job", "职位", "岗位", "招聘")):
        cand_jd = suf.strip()
    elif len((pfx or "").strip()) >= 120:
        cand_jd = (pfx or "").strip()
    if len(cand_jd) >= 120:
        session["jd_sticky_plain"] = cand_jd[:25000]


def should_trigger_resume_gap_analysis(
    user_full_text: str,
    *,
    resume_plain_len: int,
    jd_plain_len: int,
    intent_resume: bool,
) -> bool:
    """
    是否在本次轮次触发 Gap 工具。规则偏保守：须有足够文本长度 +（意图或触发词）。
    """
    t = user_full_text or ""
    trig = ("gap" in t.lower())
    trig = trig or any(
        x in t
        for x in (
            "匹配度",
            "对标",
            "短板",
            "差距",
            "简历和jd",
            "简历跟jd",
            "简历与jd",
            "差距分析",
            "能不能过",
            "够不够格",
        )
    )
    if intent_resume and any(
        x in t
        for x in (
            "gap",
            "GAP",
            "对标",
            "匹配",
            "差距",
            "短板",
            "jd",
            "JD",
            "岗位",
        )
    ):
        trig = True
    if not trig:
        return False
    return resume_plain_len >= 80 and jd_plain_len >= 120


def resolve_jd_plain_for_gap(
    user_full_text: str,
    session: dict[str, Any] | None,
) -> str:
    """
    拼装用于 Gap 分析的 JD 正文：
    - 本条消息标记前的长文本（常为粘贴 JD）；
    - 或侧栏文件名像 JD 时，本条「上传段落」视作 JD；
    - 或使用会话 jd_sticky_plain（上一波 JD 分析/粘贴缓存）。
    """
    prefix, upload_tail = split_user_message_and_upload_content(user_full_text)
    fname = str((session or {}).get("last_upload_name") or "").lower()

    if upload_tail and any(k in fname for k in ("jd", "job", "职位", "岗位", "招聘")):
        return upload_tail.strip()[:20000]

    if len(prefix.strip()) >= 120:
        return prefix.strip()[:20000]

    if isinstance(session, dict):
        sticky = (session.get("jd_sticky_plain") or "").strip()
        if len(sticky) >= 120:
            return sticky[:20000]

    return (prefix.strip() or "")[:20000]


def resolve_resume_plain_for_gap(
    user_full_text: str,
    session: dict[str, Any] | None,
) -> str:
    """
    简历正文：优先侧栏上传缓存；若没有，则本条上传块在「非 JD 文件名」时可作为简历兜底。
    """
    r = read_resume_plain_from_session(session)
    if len(r) >= 80:
        return r[:20000]

    prefix, upload_tail = split_user_message_and_upload_content(user_full_text)
    fname = str((session or {}).get("last_upload_name") or "").lower()
    jd_name = upload_tail and any(k in fname for k in ("jd", "job", "职位", "岗位", "招聘"))
    if jd_name:
        stale = ""
        if isinstance(session, dict):
            stale = (session.get("resume_sticky_plain") or "").strip()
        return stale[:20000] if len(stale) >= 80 else r

    if len(upload_tail.strip()) >= 80:
        return upload_tail.strip()[:20000]
    _ = prefix
    return r[:20000]


# ---------------------------------------------------------------------------
# RAG（门面，调用 rag_engine，避免多处直接 import rag_engine）
# ---------------------------------------------------------------------------
def rag_search_tools(query: str, top_k: int | None = None) -> list[str]:
    """
    从本地向量库检索与 query 最接近的文本块（与 rag_engine.rag_search 一致）。

    top_k：默认沿用 config.RAG_TOP_K。
    """
    from rag_engine import rag_search as _rag_search

    return _rag_search(query, top_k=top_k)


# ---------------------------------------------------------------------------
# 模拟面试：状态键名与只读快照（面试官状态机在 agent_brain）
# ---------------------------------------------------------------------------
MOCK_SESSION_KEYS_ACTIVE = "mock_interview_active"
MOCK_SESSION_KEYS_STAGE_INDEX = "mock_emit_index"
MOCK_SESSION_KEYS_TRANSCRIPT = "mock_transcript"


def mock_interview_is_active(session: dict[str, Any] | None) -> bool:
    if not isinstance(session, dict):
        return False
    return bool(session.get(MOCK_SESSION_KEYS_ACTIVE))


def mock_interview_snapshot(session: dict[str, Any]) -> dict[str, Any]:
    """
    调试 / UI：返回只读快照，不向 session 写入。
    """
    if not isinstance(session, dict):
        return {}
    return {
        "active": bool(session.get(MOCK_SESSION_KEYS_ACTIVE)),
        "awaiting_answer": bool(session.get("mock_awaiting_answer")),
        "next_emit_index": session.get(MOCK_SESSION_KEYS_STAGE_INDEX),
        "resume_digest_len": len((session.get("mock_resume_digest") or "")),
        "completed_rounds": len(session.get(MOCK_SESSION_KEYS_TRANSCRIPT) or []),
    }


# ---------------------------------------------------------------------------
# 文档解析（简历/JD）
# ---------------------------------------------------------------------------
def extract_plain_from_upload(name: str, raw_bytes: bytes) -> str:
    """从上传文件解析纯文本。"""
    lower = (name or "").lower()
    if lower.endswith(".pdf"):
        return _read_pdf_text(raw_bytes)
    if lower.endswith((".md", ".markdown", ".txt")):
        return raw_bytes.decode("utf-8", errors="replace")
    return raw_bytes.decode("utf-8", errors="replace")


def _read_pdf_text(raw_bytes: bytes) -> str:
    import io

    try:
        import pdfplumber
    except ImportError as exc:
        logger.warning("未安装 pdfplumber: %s", exc)
        return "[PDF] 需要安装 pdfplumber 才能解析 PDF。"

    try:
        parts: list[str] = []
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                parts.append(t)
        return "\n".join(parts).strip() or "[PDF] 未提取到文本，可能是扫描件。"
    except Exception as exc:  # noqa: BLE001
        logger.exception("解析 PDF 失败: %s", exc)
        return f"[PDF] 解析失败: {exc}"
