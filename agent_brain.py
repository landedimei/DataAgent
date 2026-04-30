# -*- coding: utf-8 -*-
"""
agent_brain.py — 核心控制层：意图路由 + 轻量 ReAct/工具决策 + OpenAI 兼容调用

你后续会主要在这个文件里扩展：
- 更完整的 ReAct 循环（多步规划、多工具链）
- 与 rag_engine、mock 面试控制器的配合

为便于学习，本文件用「一个类 + 几个纯函数」组织，数据流是线性的，便于下断点调试。
"""
from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, Mapping

from openai import APITimeoutError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config
from rag_engine import is_vector_store_empty, rag_search
from tools import (
    analyze_jd,
    analyze_resume_gap,
    read_resume_plain_from_session,
    resolve_jd_plain_for_gap,
    resolve_resume_plain_for_gap,
    should_trigger_resume_gap_analysis,
    sync_gap_sticky_caches,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 意图标签：与需求文档中的主路由一致
# ---------------------------------------------------------------------------
class UserIntent(str, Enum):
    """用户意图（路由结果）。"""

    KNOWLEDGE = "knowledge"  # 知识问答
    JD_ANALYSIS = "jd_analysis"  # 想做 JD 分析
    RESUME_UPLOAD = "resume_upload"  # 与简历/上传相关
    MOCK_INTERVIEW = "mock_interview"  # 模拟面试
    GENERAL = "general"  # 闲聊/其它


class QwenClient:
    """
    封装「通义千问 + OpenAI 官方 Python SDK 兼容模式」的聊天调用。

    数据流简图：
    你写的 system/user 提示词 → OpenAI 客户端（指向 DashScope 的 compatible URL）
    → 通义服务 → 返回的 choices[0].message.content 字符串
    """

    def __init__(self) -> None:
        if not config.DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY 未设置，请配置 .env")
        # base_url 指向「兼容 OpenAI 协议」的网关；api_key 用 DashScope 的 Key
        self._client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.QWEN_BASE_URL,
        )
        self._model = config.QWEN_CHAT_MODEL

    # 仅对「超时类」重试，避免 401/403 等鉴权问题被无意义地连打
    @retry(
        stop=stop_after_attempt(config.LLM_MAX_RETRIES + 1),  # 第 1 次 + 最多重试 3 次
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((APITimeoutError, TimeoutError)),
        reraise=True,
    )
    def _chat_once(self, messages: list[ChatCompletionMessageParam]) -> str:
        """
        单次非流式聊天。失败时由 tenacity 按指数退避重试（受 retry 条件限制）。

        为什么用 ChatCompletionMessageParam：
        - 与 openai 库期望的「角色 + 内容」结构一致，类型检查器不会误报。
        """
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        choice = resp.choices[0]
        if not choice.message or not choice.message.content:
            raise ValueError("模型返回空内容")
        return choice.message.content.strip()

    @retry(
        stop=stop_after_attempt(config.LLM_MAX_RETRIES + 1),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((APITimeoutError, TimeoutError)),
        reraise=True,
    )
    def create_chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> Any:
        """
        通用聊天补全（可带 tools）。

        **供 Tool Agent**：返回完整 response，调用方读取 choices[0].message（含可选 tool_calls）。
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.3,
            "timeout": config.LLM_TIMEOUT_SECONDS,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        return self._client.chat.completions.create(**kwargs)

    def simple_chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        对外最简 API：一个 system + 一个 user，返回助手文本。

        这是整个 Agent 的「大模型出口」，意图分类、ReAct 推理、总结评价都会走这里或类似封装。
        """
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._chat_once(messages)

    def classify_intent(self, user_text: str, last_intent: str | None) -> str:
        """
        用零样本提示让模型从固定标签里选一个；解析失败时返回 general，交由上层用关键词纠偏。

        为什么不用复杂输出 JSON Schema：
        - 你当前学习阶段，先看「纯文本 JSON」更容易调试；若模型吐垃圾，有兜底。
        """
        system = (
            "你是文本分类器。只输出一个 JSON 对象，不要解释。"
            f'键 intent 的取值必须是：{", ".join([e.value for e in UserIntent])}。'
            f"如果用户要模拟面试、考察面试题，用 mock_interview；"
            f"如果主要粘贴或讨论岗位描述/JD/招聘要求，用 jd_analysis；"
            f"如果涉及简历/上传/个人经历匹配，用 resume_upload；"
            f"如果问数仓/Spark/SQL/DE 等知识，用 knowledge；否则 general。"
        )
        if last_intent:
            system += f" 上一轮意图是 {last_intent}，可作参考但不要盲从。"
        user = f"用户输入：\n{user_text[:8000]}"
        raw = self.simple_chat(system, user)
        return _parse_intent_json(raw) or "general"


def _parse_intent_json(raw: str) -> str | None:
    """
    从模型返回里抠出 intent 字符串；若模型多打了字，用粗暴正则/ json 都尝试一下。
    """
    raw = raw.strip()
    # 有时模型会包在 ```json ... ``` 里
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "intent" in data:
            val = str(data["intent"]).strip()
            if val in {e.value for e in UserIntent}:
                return val
    except json.JSONDecodeError:
        pass
    # 退而求其次：在文本里找第一个合法标签
    for e in UserIntent:
        if e.value in raw:
            return e.value
    return None


# ---------------------------------------------------------------------------
# 关键词兜底：当 LLM 分类异常或不想调用模型时，用规则保一条明确路径
# ---------------------------------------------------------------------------
def keyword_intent_fallback(user_text: str) -> UserIntent:
    t = (user_text or "").lower()
    if any(k in user_text for k in ("模拟面试", "面试模式", "当面试官", "考我")) or (
        "面试" in user_text and ("开始" in user_text or "来" in user_text)
    ):
        return UserIntent.MOCK_INTERVIEW
    if any(
        k in user_text
        for k in (
            "简历",
            "我的经历",
            "上传",
            "附件",
        )
    ):
        return UserIntent.RESUME_UPLOAD
    if "jd" in t or "岗位描述" in user_text or "招聘" in user_text and "要求" in user_text:
        return UserIntent.JD_ANALYSIS
    return UserIntent.GENERAL


def route_intent(
    user_text: str, llm: QwenClient | None, last_intent: str | None = None
) -> UserIntent:
    """
    主路由：优先 LLM 零样本，失败时关键词兜底。保证不会因为异常而卡死。
    用llm大模型判断用户提问是哪个路由
    教学要点：生产里常见「主模型 + 规则兜底」，规则成本低、可预期。
    """
    if llm is not None:
        try:
            label = llm.classify_intent(user_text, last_intent)
            if label in {e.value for e in UserIntent}:
                return UserIntent(label)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM 意图分类失败，使用关键词兜底: %s", exc)
    return keyword_intent_fallback(user_text)


def _intent_allows_rag(intent: UserIntent) -> bool:
    """
    在这些意图下尝试 RAG。

    - 知识/闲聊：检索知识点 + 可含面经里的考点描述。
    - 模拟面试：同样走检索——知识库里的「面经」能帮你对齐真题风格、针对性出题。
    不做 RAG 的：JD 专项（走 analyze 工具链）、纯简历分析（可后续加 resume+JD 对拍）。
    """
    return intent in (
        UserIntent.KNOWLEDGE,
        UserIntent.GENERAL,
        UserIntent.MOCK_INTERVIEW,
    )


def _build_rag_search_query(
    current_user_text: str, session: Mapping[str, Any] | None
) -> str:
    """
    供向量检索用的一句话/小段（不是最终给用户的答案）。
    这里就是对user_prompt进行增强，和历史对话进行拼接，加强模型理解
    多轮对话里用户常说「那上面那个呢」「再细一点」—— 若只用当前句做 embedding，容易搜偏；
    因此把「上一轮用户原话」与本轮拼在一起再检索，符合需求里「基于会话历史扩展检索上下文」。
    """
    cur = (current_user_text or "").strip()
    if not cur:
        return cur
    if not isinstance(session, dict):
        return cur
    prev = (session.get("last_user_message") or "").strip()
    if not prev or prev == cur:
        return cur
    # 限制长度，避免 embedding 单条过长被拒
    tail = f"{prev[:800]}\n---\n{cur}"
    return tail[:2000]


def _format_rag_context(chunks: list[str]) -> str:
    """
    把 Chroma 返回的若干段文字拼成「模型可读的一块 context」。

    用编号分块，模型在回答时可以说「据片段1…」，也便于你调试检索质量。
    """
    parts: list[str] = []
    for i, text in enumerate(chunks, 1):
        t = (text or "").strip()
        if t:
            parts.append(f"### 片段 {i}\n{t}")
    return "\n\n".join(parts) if parts else ""


RAG_SYSTEM_PREFIX = (
    "你是数据开发（DE）方向的面试辅导助手。下面「参考资料」来自本地题库的检索结果，"
    "请你**优先**依据这些片段作答；若片段不足以支撑结论，可补充你掌握的行业通用知识，"
    "并简要说明「资料中未直接涉及」的部分。回答用中文，条理清晰，可使用列表。"
)

RAG_SYSTEM_WITH_CONTEXT = (
    RAG_SYSTEM_PREFIX
    + "\n\n## 参考资料（检索得到，可能不完整，请勿编造片段中不存在的原话）\n\n{context}\n"
)

# 模拟面试：语料中常混有「面经、真题风格、考察点」——专用 prompt 引导模型以面试官身份、对齐面经出题
RAG_MOCK_INTERVIEW_PREFIX = (
    "你正在扮演**数据开发（DE）方向的技术面试官**。"
    "下面「参考资料」来自本地知识库检索，其中可能包含**知识点、面经、真题或考点描述**；"
    "请**优先**根据这些材料中的问法、考点、场景来**向用户出题或续问下一题**（可要求口述思路、手撕 SQL/架构等，按材料贴近度选）。"
    "若某段材料与当前轮次弱相关，可略过，结合通用 DE 面试规范提问。全程中文，可简短说明本题的考察点。"
)

RAG_MOCK_INTERVIEW_WITH_CONTEXT = (
    RAG_MOCK_INTERVIEW_PREFIX
    + "\n\n## 参考资料（含面经/知识/考点片段）\n\n{context}\n"
)

# 用户只说「开始面试」时，纯向量召回可能偏虚；在检索用 query 后追加轻量词，拉齐与「面经/考点」的语义距离
_MOCK_RAG_RETRIEVAL_HINT = "数据开发 面试 面经 常考 技术题"


def _maybe_boost_mock_search_query(search_query: str, intent: UserIntent) -> str:
    """
    模拟面试场景下，给检索用 query 加一小段「面经/考点」向提示，不写入给用户的最终 prompt，只影响召回。
    """
    if intent != UserIntent.MOCK_INTERVIEW:
        return search_query
    q = (search_query or "").strip()
    if not q:
        return _MOCK_RAG_RETRIEVAL_HINT
    combined = f"{q}\n{_MOCK_RAG_RETRIEVAL_HINT}"
    return combined[:2000]


def _rag_system_template_for_intent(intent: UserIntent) -> str:
    """knowledge / general 用答疑模板；模拟面试用面试官+面经模板。"""
    if intent == UserIntent.MOCK_INTERVIEW:
        return RAG_MOCK_INTERVIEW_WITH_CONTEXT
    return RAG_SYSTEM_WITH_CONTEXT


def _answer_with_rag_grounding(
    user_text: str,
    llm: QwenClient,
    search_query: str,
    *,
    used_rag: list[bool],
    intent: UserIntent,
    user_message_override: str | None = None,
) -> str | None:
    """
    若向量库有内容且检索到块，则拼 prompt 并返回回复；无法走 RAG 时返回 None，由调用方走纯 LLM。

    使用 list[used_rag] 做「出参」：Streamlit 侧可据 session[used_rag] 在 UI 上打「已参考知识库」。

    user_message_override: 多轮模拟面试时传入整段「历史+指令」；为 None 时仅使用 user_text。
    """
    if is_vector_store_empty():
        logger.info("RAG: 知识库为空，跳过检索")
        return None

    chunks = rag_search(search_query, top_k=config.RAG_TOP_K)
    if not chunks:
        logger.info("RAG: 未检索到相关块，将降级为纯 LLM")
        return None

    used_rag.append(True)
    context = _format_rag_context(chunks)
    system_tpl = _rag_system_template_for_intent(intent)
    system = system_tpl.format(context=context)
    body = (user_message_override if user_message_override is not None else (user_text or "")).strip()
    if intent == UserIntent.MOCK_INTERVIEW:
        user_block = f"## 用户当前话轮（面试进行中）\n{body}"
    else:
        user_block = f"## 用户当前问题\n{body}"
    return llm.simple_chat(system, user_block)


# ---------------------------------------------------------------------------
# 模拟面试状态机：基于简历的分阶段面试 → 终局评价后 idle
#
# 一轮面试固定顺序（每步一次「面试官出题」+ 候选人作答）：
# 1) 项目深挖（实习/项目：做什么、难点、结合技术追问）
# 2) 技术八股 + 场景题（可结合简历技术栈 + RAG 面经）
# 3) 开放性题
# 4) 手写 SQL 题（给表结构/需求）
# 5) 力扣风格算法题（中等偏易）
#
# session 扩展键：mock_emit_index, mock_resume_plain, mock_resume_digest
# ---------------------------------------------------------------------------
MOCK_PIPELINE: tuple[str, ...] = (
    "project_deep",  # 简历项目/实习深挖
    "tech_scenario",  # 八股 + 场景
    "open_ended",  # 开放题
    "sql_coding",  # 一道 SQL
    "algo_coding",  # 一道算法题
)

MOCK_ALL_STAGES_DONE_HINT = (
    "你已经完成本轮模拟面试中的全部环节\n"
    "请回复 **「请总结」** 或 **「结束面试」** 获取综合评价报告。"
)


def _load_resume_plain_from_session(session: dict[str, Any]) -> str:
    """从侧栏上传的 last_upload 解析简历文本；供模拟面试专用（与 Gap 共用 tools 解析）。"""
    try:
        return read_resume_plain_from_session(session).strip()
    except Exception as exc:  # noqa: BLE001
        logger.exception("解析简历失败: %s", exc)
        return ""


def _summarize_resume_for_mock(llm: QwenClient, plain: str) -> str:
    """
    把长简历压成「项目/实习要点」，减少 token、方便模型挑点追问。

    若调用失败则退回截断原文。
    """
    if not plain or len(plain) < 80:
        return plain[:4000] if plain else ""
    system = (
        "你是招聘助理。请**只根据**下面简历内容，用中文整理成结构化要点，"
        "不得编造简历没有的信息。包含：\n"
        "1) 教育/工作概览（几行内）\n"
        "2) 实习与项目经历（每条：名称/时间若存在、你负责什么、技术栈关键词、有哪些项目亮点难点、可追问点）\n"
        "3) 若几乎无项目，说明「项目信息较少」\n"
        "总字数不超过 900 字。"
    )
    try:
        return llm.simple_chat(system, f"简历全文：\n{plain[:14000]}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("简历 digest 失败，使用截断原文: %s", exc)
        return plain[:3500]


def _build_mock_stage_instruction_block(
    stage: str,
    *,
    resume_digest: str,
    resume_excerpt: str,
    history: str,
) -> str:
    """
    按当前阶段拼出给大模型的「任务说明」块（再交给 RAG/LLM 出题）。

    resume_digest：要点提炼；resume_excerpt：原文截断备份，避免 digest 漏信息。
    """
    base_resume = (
        "## 候选人简历要点（务必结合提问，勿编造）\n"
        f"{resume_digest or '（未提供简历要点，请要求候选人先上传简历或口述一段项目经历。）'}\n\n"
        "## 简历原文摘录（节选，供核对）\n"
        f"{resume_excerpt[:6000] if resume_excerpt else '（无）'}\n"
    )
    hist = f"## 已进行的面试记录\n{history}\n"

    if stage == "project_deep":
        return (
            base_resume
            + hist
            + "## 本阶段任务：项目 / 实习深挖\n"
            "请针对简历中的**实习、项目、科研或课程大作业**中**最相关数据/工程经历**提问（若多条可优先最近或最相关的一条）。\n"
            "一次输出中可包含一个**主问题**和 1～2 个**追问方向**，覆盖：项目目标与背景、你**具体负责**的工作、**技术难点**与如何克服、"
            "若合适再结合项目技术栈追问**原理**（如 Spark/Hive/Flink/SQL/数据质量等）。\n"
            "语气为现场技术面试官，中文；若简历无项目，请礼貌请候选人**口述一段可代表经历**再追问。"
        )

    if stage == "tech_scenario":
        return (
            base_resume
            + hist
            + "## 本阶段任务：技术八股 + 场景题\n"
            "请出 3～4 道**数据开发方向**常见八股（如数仓分层、Spark/Flink、一致性、SQL 优化、数据倾斜等），"
            "并尽量结合**简历中出现的技术栈**；再加 1~2 道**贴近业务的场景题**（如何设计、如何排查、权衡点）。\n"
        )

    if stage == "open_ended":
        return (
            base_resume
            + hist
            + "## 本阶段任务：开放性问题\n"
            "请出 **1 道**开放性题目，可从：技术视野与成长、跨团队协作、需求冲突处理、"
            "对数据驱动业务的理解、职业规划与动机等中选一条，贴近 DE 岗位。中文。"
        )

    if stage == "sql_coding":
        return (
            base_resume
            + hist
            + "## 本阶段任务：SQL 编程题\n"
            "请出 **一道**可在面试中手写完成的 **SQL 题**：用 Markdown 给出**简化表结构**（或文字描述列含义）、"
            "**业务需求**、**期望输出说明**；难度参考中高级数据开发 SQL 笔试；与算法题独立、不混在一题里。中文。"
        )

    if stage == "algo_coding":
        return (
            base_resume
            + hist
            + "## 本阶段任务：算法题（力扣风格）\n"
            "请出 **一道**算法题，难度约 **LeetCode 中等偏易** / 剑指 offer 常见题：写清**题目描述**、**输入输出格式**、"
            "**约束与 1～2 个样例**；说明可用伪代码或任意常见语言思路。题目须与上一道 SQL **业务上独立**，不要重复 SQL 场景。中文。"
        )

    return base_resume + hist + f"## 未知阶段 {stage}\n请继续技术面试。"


def _is_mock_end_request(ut: str) -> bool:
    """长文本视为作答，不当作结束语；仅短句里的明确结束/总结意图触发终局评价。"""
    t = (ut or "").strip()
    if not t or len(t) > 200:
        return False
    phrases = (
        "结束面试",
        "结束吧",
        "不面了",
        "不考了",
        "不继续了",
        "请总结",
        "给我总结",
        "给我评价",
        "出评价",
        "请评价",
        "面试结束",
        "退出面试",
        "先到这里",
        "不用继续",
        "不要面试了",
    )
    return any(p in t for p in phrases)


def _keyword_wants_mock_start(ut: str) -> bool:
    """在意图分类偶尔失误时，用语义关键词兜底为「要开模拟面」。"""
    t = (ut or "").strip()
    if len(t) > 100:
        return False
    keys = (
        "开始模拟面试",
        "模拟面试开始",
        "开始面试",
        "来场模拟",
        "当面试官",
        "考我一道",
        "面试考我",
    )
    return any(k in t for k in keys)


def _format_completed_mock_rounds(session: dict[str, Any]) -> str:
    """仅已落盘的「题+答」轮次，供续问下一题时拼进 prompt，不含未作答的尾题。"""
    rows: list[dict[str, str]] = session.get("mock_transcript") or []
    parts: list[str] = []
    for i, row in enumerate(rows, 1):
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        parts.append(f"### 第 {i} 题（面试官）\n{q}\n\n### 第 {i} 题（你）\n{a}\n")
    return "\n".join(parts) if parts else "（尚无任何已答题目。）\n"


def _format_mock_transcript_for_eval(session: dict[str, Any]) -> str:
    """把已记录的问答 + 可能未答的尾题，格式化为终局评价用的纯文本。"""
    rows: list[dict[str, str]] = session.get("mock_transcript") or []
    if not rows and not (session.get("mock_last_question") or "").strip():
        return "（用户尚未与面试官完成任何有记录的问答。）\n"
    parts: list[str] = []
    for i, row in enumerate(rows, 1):
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        parts.append(f"### 第 {i} 题（面试官）\n{q}\n\n### 第 {i} 题（候选人作答）\n{a}\n")
    # 若面试官已发出最后一题但用户直接说「结束」而未作答，用 pending 题补全信息
    pending = (session.get("mock_last_question") or "").strip()
    if not pending:
        return "\n".join(parts) if parts else "（无记录。）\n"
    last_q_in_rows = (rows[-1].get("question") or "").strip() if rows else ""
    is_same_as_last_recorded = bool(last_q_in_rows) and (pending in last_q_in_rows or last_q_in_rows in pending)
    if not is_same_as_last_recorded and len(pending) > 1:
        parts.append("### 已发出但候选人未作答的追问/题目\n" + pending + "\n")
    return "\n".join(parts) if parts else "（无记录。）\n"


def _reset_mock_interview_state(session: dict[str, Any]) -> None:
    for k in (
        "mock_interview_active",
        "mock_awaiting_answer",
        "mock_last_question",
        "mock_transcript",
        "mock_emit_index",
        "mock_resume_plain",
        "mock_resume_digest",
    ):
        session.pop(k, None)


def _run_mock_rag_or_llm(
    user_text: str,
    user_message_override: str,
    search_query: str,
    llm: QwenClient,
    session: dict[str, Any],
) -> str:
    """
    模拟面试里统一：先试 RAG（面经/考点），再退回面试官风格纯 LLM。
    """
    used: list[bool] = []
    try:
        r = _answer_with_rag_grounding(
            user_text,
            llm,
            search_query,
            used_rag=used,
            intent=UserIntent.MOCK_INTERVIEW,
            user_message_override=user_message_override,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("模拟面试 RAG 异常: %s", exc)
        r = None
    if r is not None:
        session["last_reply_used_rag"] = bool(used)
        return r
    session["last_reply_used_rag"] = False
    system = (
        "你是数据开发（DE）方向的技术面试官。下面是对话/指令的完整块，"
        "请按其中要求**出题、追问或点评**（以中文），贴近真实技术面试。若可引用通用考点也可简述。"
    )
    return llm.simple_chat(system, user_message_override)


def _mock_start_interview(
    user_text: str, session: dict[str, Any], llm: QwenClient
) -> str:
    """开局：载入简历→digest→按 MOCK_PIPELINE 第 1 阶段（项目深挖）出题。"""
    session["mock_interview_active"] = True
    session["mock_awaiting_answer"] = False
    session["mock_transcript"] = []

    plain = _load_resume_plain_from_session(session)
    session["mock_resume_plain"] = plain[:20000]
    session["mock_resume_digest"] = _summarize_resume_for_mock(llm, plain) if plain.strip() else ""

    # emit_index：下一题将使用 pipeline[self.mock_emit_index]；首开已出第 0 题后设为 1
    session["mock_emit_index"] = 1
    stage0 = MOCK_PIPELINE[0]
    history = _format_completed_mock_rounds(session)
    block = _build_mock_stage_instruction_block(
        stage0,
        resume_digest=session.get("mock_resume_digest") or "",
        resume_excerpt=plain[:12000],
        history=history,
    )
    if not plain.strip():
        block = (
            "【系统提示】侧栏暂未解析到有效简历正文。你作为面试官：先请候选人用 3～5 句话口述一段"
            "「项目或实习」经历（背景、职责、技术、难点），再按下面「项目深挖」要求追问。\n\n"
        ) + block

    header = (
        f"候选人刚表示希望开始 DE 模拟面试。用户触发语：「{user_text.strip()[:800]}」\n"
        "---\n\n"
    )
    s_q = _maybe_boost_mock_search_query(_build_rag_search_query(user_text, session), UserIntent.MOCK_INTERVIEW)
    reply = _run_mock_rag_or_llm(user_text, header + block, s_q, llm, session)
    session["mock_last_question"] = reply
    session["mock_awaiting_answer"] = True
    return reply


def _mock_continue_interview(
    user_text: str, session: dict[str, Any], llm: QwenClient
) -> str:
    """用户答完上一轮后：按 MOCK_PIPELINE 进入下一阶段出题；全部结束后提示去要总结。"""
    last_q = (session.get("mock_last_question") or "").strip()
    if last_q:
        trans: list[dict[str, str]] = list(session.get("mock_transcript") or [])
        trans.append({"question": last_q, "answer": (user_text or "").strip()})
        session["mock_transcript"] = trans
    session["mock_last_question"] = ""

    emit_idx = int(session.get("mock_emit_index", 0))
    if emit_idx >= len(MOCK_PIPELINE):
        return MOCK_ALL_STAGES_DONE_HINT

    stage = MOCK_PIPELINE[emit_idx]
    history = _format_completed_mock_rounds(session)
    plain = (session.get("mock_resume_plain") or "")[:20000]
    digest = (session.get("mock_resume_digest") or "")[:10000]
    block = _build_mock_stage_instruction_block(
        stage,
        resume_digest=digest,
        resume_excerpt=plain[:12000],
        history=history,
    )
    if not plain.strip():
        block = "【简历仍为空或未解析】请以候选人上一轮**口述经历与答题**为依据继续本节任务。\n\n" + block

    # 本步将发出 pipeline[emit_idx]，下轮应从 emit_idx+1 开始
    session["mock_emit_index"] = emit_idx + 1

    lead = (
        f"请在输出开头用一行小标题标注当前环节，例如："
        f"「**本节：{_stage_cn_name(stage)}**」，然后正式出题。\n---\n\n"
    )
    s_q = _maybe_boost_mock_search_query(
        _build_rag_search_query(user_text, session), UserIntent.MOCK_INTERVIEW
    )
    reply = _run_mock_rag_or_llm(user_text, lead + block, s_q, llm, session)
    session["mock_last_question"] = reply
    session["mock_awaiting_answer"] = True
    return reply


def _stage_cn_name(stage: str) -> str:
    mapping = {
        "project_deep": "项目/实习深挖",
        "tech_scenario": "技术八股与场景题",
        "open_ended": "开放性问题",
        "sql_coding": "SQL 手写题",
        "algo_coding": "算法题（力扣风格）",
    }
    return mapping.get(stage, stage)


def _mock_final_evaluation_and_reset(
    user_text: str, session: dict[str, Any], llm: QwenClient
) -> str:
    body = _format_mock_transcript_for_eval(session)
    _reset_mock_interview_state(session)
    system = (
        "你是资深数据开发（DE）面试官/辅导老师。请根据下面「模拟面试过程」的问答记录，"
        "用中文写一份**综合评价**；若记录极少或没有实质作答，请说明样本有限并给通用学习建议，不要编造不存在的答句。"
        "报告结构需包含这些 Markdown 二级标题：## 整体印象、## 技术深度与知识点、## 表达与逻辑、## 改进建议。"
    )
    user = f"用户表示结束本次模拟面试。完整问答记录：\n\n{body}\n"
    try:
        return llm.simple_chat(system, user)
    except Exception as exc:  # noqa: BLE001
        logger.exception("终局评价失败: %s", exc)
        return "抱歉，生成评价报告时暂时失败，请稍后再试。你可以再说一次「请总结」。"


def _try_mock_interview_state_machine(
    user_text: str,
    session: Mapping[str, Any],
    llm: QwenClient,
    intent: UserIntent,
) -> str | None:
    """
    若本话轮由「模拟面试状态机」处理，返回 str；若应交给后面的通用 RAG/LLM，返回 None。

    状态：未开局且 intent/关键词 为开面 → 第一题；已开局且等答 → 落盘一轮并下一题；短句结束 → 终局报告。
    """
    if not isinstance(session, dict):
        return None

    active = bool(session.get("mock_interview_active"))
    ut = (user_text or "").strip()
    if not ut and not _is_mock_end_request(ut):
        return None

    if active and _is_mock_end_request(ut):
        if isinstance(session, dict):
            session["last_reply_used_rag"] = False
        _session_note_last_user(session, user_text)
        return _mock_final_evaluation_and_reset(user_text, session, llm)

    if active and session.get("mock_awaiting_answer"):
        if isinstance(session, dict):
            session["last_reply_used_rag"] = False
        _session_note_last_user(session, user_text)
        return _mock_continue_interview(user_text, session, llm)

    if not active and (intent == UserIntent.MOCK_INTERVIEW or _keyword_wants_mock_start(ut)):
        if isinstance(session, dict):
            session["last_reply_used_rag"] = False
        _session_note_last_user(session, user_text)
        return _mock_start_interview(user_text, session, llm)

    return None


def should_use_jd_tool(user_text: str, intent: UserIntent) -> bool:
    """
    轻量「工具使用」判断：与需求中「当输入含 JD 时触发 analyze_jd 原型」对齐。

    这还不是完整 ReAct，只演示「if 需要工具 → 调函数」的骨架。
    """
    t = (user_text or "").lower()
    if "jd" in t or "岗位" in user_text and "分析" in user_text:
        return True
    if intent == UserIntent.JD_ANALYSIS and len(user_text.strip()) > 20:
        return True
    return False


def run_agent_turn(
    user_text: str,
    session: Mapping[str, Any],
    llm: QwenClient,
) -> str:
    """
    单轮主流程（供 Streamlit 调用）：

    1. 读 session 里上一次的 intent、last_user_message（供 RAG 多轮拼检索 query）
    2. 路由出当前 intent
    3. 若应跑 JD 工具，则 `analyze_jd` 后返回
    4. 若命中模拟面试状态机（开局 / 续问 / 终局评价），由该段返回
    5. 对 knowledge / general 尝试 RAG；mock 仅走状态机内的 `_run_mock_rag_or_llm`
    6. 否则走纯 LLM 降级
    """
    last = session.get("last_intent") if isinstance(session, Mapping) else None
    intent = route_intent(user_text, llm, last)
    if isinstance(session, dict):
        session["last_intent"] = intent.value
        sync_gap_sticky_caches(session, user_text)

    if (not config.USE_TOOL_AGENT_LOOP) and should_use_jd_tool(
        user_text, intent
    ) and len((user_text or "").strip()) > 10:
        if isinstance(session, dict):
            session["last_reply_used_rag"] = False
        _session_note_last_user(session, user_text)
        # 用 LLM 增强 analyze_jd；若你希望先稳定跑通，可改为 use_llm=False
        return analyze_jd(
            user_text,
            use_llm=True,
            llm_caller=llm,
        )

    # ---------- 模拟面试状态机（多轮题/答、RAG+面经、终局评价）— 在通用 RAG 之前处理 ----------
    if isinstance(session, dict):
        m_out = _try_mock_interview_state_machine(user_text, session, llm, intent)
        if m_out is not None:
            return m_out

    # ---------- 简历 vs JD Gap（须在模拟面试之后，以免打断面试流程）----------
    if (not config.USE_TOOL_AGENT_LOOP) and isinstance(session, dict):
        rp = resolve_resume_plain_for_gap(user_text, session)
        jp = resolve_jd_plain_for_gap(user_text, session)
        if should_trigger_resume_gap_analysis(
            user_text,
            resume_plain_len=len(rp.strip()),
            jd_plain_len=len(jp.strip()),
            intent_resume=(intent == UserIntent.RESUME_UPLOAD),
        ):
            session["last_reply_used_rag"] = False
            _session_note_last_user(session, user_text)
            return analyze_resume_gap(
                rp, jp, use_llm=True, llm_caller=llm
            )

    # ---------- OpenAI 风格「工具自主决策」多轮循环（替代下方线性 JD/Gap/RAG）----------
    if config.USE_TOOL_AGENT_LOOP and isinstance(session, dict):
        from tool_agent import run_tool_agent_turn

        _session_note_last_user(session, user_text)
        return run_tool_agent_turn(user_text, session, llm)

    # ---------- RAG 分支：仅 knowledge / general；mock_interview 已完全由状态机 + _run_mock_rag_or_llm 处理 ----------
    # used_rag 用列表作可变标记，供 app 侧展示「本轮是否参考了知识库」
    used_rag_flag: list[bool] = []
    if _intent_allows_rag(intent) and intent != UserIntent.MOCK_INTERVIEW:
        s_q = _build_rag_search_query(user_text, session)
        s_q = _maybe_boost_mock_search_query(s_q, intent)
        try:
            rag_reply = _answer_with_rag_grounding(
                user_text, llm, s_q, used_rag=used_rag_flag, intent=intent
            )
        except Exception as exc:  # noqa: BLE001 — 检索/模型任一环失败都降级
            logger.exception("RAG 流程异常，降级纯 LLM: %s", exc)
            rag_reply = None
        if rag_reply is not None:
            if isinstance(session, dict) and used_rag_flag:
                session["last_reply_used_rag"] = True
            _session_note_last_user(session, user_text)
            return rag_reply

    # ---------- 无检索命中 / 不启用 RAG / 或知识库为空：需求中的「纯 LLM 降级」----------
    if isinstance(session, dict):
        session["last_reply_used_rag"] = False
    if intent == UserIntent.MOCK_INTERVIEW:
        # 未命中面经/语料时仍以「面试官」身份提问，避免退化成普通答疑口吻
        system = (
            "你是数据开发（DE）方向的技术面试官，正在与用户进行模拟面试。"
            "当前未命中本地知识库片段，请仍用贴近真实技术面试的方式出题、追问或点评（视用户上一句而定），中文回复。"
        )
    else:
        system = (
            "你是数据开发（DE）方向的面试辅导助手，回答简洁、专业，中文回复。"
            "若用户想深入，可分点说明。"
        )
    try:
        reply = llm.simple_chat(system, user_text)
        _session_note_last_user(session, user_text)
        return reply
    except Exception as exc:  # noqa: BLE001
        logger.exception("聊天失败: %s", exc)
        return "抱歉，服务暂时不可用，请稍后再试。"


def _session_note_last_user(session: Mapping[str, Any], user_text: str) -> None:
    """
    记录「上一轮用户原话」供下一轮的检索 query 拼上下文（见 _build_rag_search_query）。
    """
    if isinstance(session, dict):
        session["last_user_message"] = (user_text or "").strip()
