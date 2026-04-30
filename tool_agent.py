# -*- coding: utf-8 -*-
"""
tool_agent.py — 基于 OpenAI 兼容协议的「工具调用」Agent 回路

工作原理（与手写 if/else 路由对照）：
1. 将若干本地能力注册为 Chat Completions 的 ``tools``（JSON Schema）。
2. 模型在单轮或多轮中选择 ``tool_choice="auto"``：可只回复文本，也可返回 ``tool_calls``。
3. 运行时在本地执行 Python 对应的工具函数，把结果以 ``role=tool`` 写回上下文，再问模型，
   直到不再有 ``tool_calls``（自然终止）或达到最大步数（安全终止）。

前置条件：**兼容端点与本聊天模型支持 tools / tool_calls**（通义 DashScope compatible-mode + qwen-plus
等通常为支持；若报错请关闭 USE_TOOL_AGENT_LOOP 并改回线性路由）。
"""
from __future__ import annotations

import json
import logging
from typing import Any

import config

from agent_brain import (
    _build_rag_search_query,
    _format_rag_context,
)

logger = logging.getLogger(__name__)

# OpenAI SDK 格式的工具定义（可被 Qwen compatible 网关接受）
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "在用户需要数仓/SQL/Spark/DE 知识、考点、面试题时使用。"
                "从本地向量检索与 query 最接近的讲义/面经片段。若用户问题短或指代「上面」可结合会话。"
                "query 应尽量覆盖用户关注点；不要用于纯闲聊。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "面向检索的中文或英文短语，可适当合并用户目标与关键技术词。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "检索条数上限，可不传则用系统默认",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_job_description",
            "description": (
                "分析岗位 JD：职责、技术要求、高频考点。**必须**在用户粘贴较长岗位描述或使用「分析JD」时出现。"
                "jd_text 应包含用户本条中的岗位正文；若本条没有足够正文，可先说明缺信息而不要编造。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {
                        "type": "string",
                        "description": "完整或尽量完整的岗位描述文本（中文/英文均可）。",
                    },
                },
                "required": ["jd_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_resume_to_jd",
            "description": (
                "对比「简历」与「岗位 JD」的匹配度与技能缺口。仅当用户明确要做对标、差距、能否胜任等"
                "且侧栏或对话中已有足够简历与 JD 信息时调用；若缺材料，工具会返回提示而不是瞎编。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resume_placeholder": {
                        "type": "string",
                        "description": "占位；传空字符串即可，简历从会话与上传缓冲解析",
                    },
                    "jd_placeholder": {
                        "type": "string",
                        "description": "占位；传空字符串即可，JD 从会话与用户话轮解析",
                    },
                },
                "required": [],
            },
        },
    },
]

AGENT_SYSTEM_PROMPT = (
    "你是数据开发（DE）方向的面试辅导助手。\n\n"
    "你可以调用工具：本地知识检索、岗位 JD 分析、简历与 JD 的缺口分析。**不要编造工具未返回的事实**。\n"
    "**决策规则**：若用户仅在闲聊、致谢、或无需外部知识即可回答一两句说清楚，可直接回复，不调工具。\n"
    "若需要事实性考点、题库内容，请调用 search_knowledge_base。\n"
    "若用户提供或讨论完整岗位 JD，需要用结构化分析时请调用 analyze_job_description，并把 jd_text 设为"
    "**用户话术中的 JD 正文**（可截断但很明确的部分）。\n"
    "若用户明确要与简历对标岗位、gap 分析，且有简历+JD材料时调用 compare_resume_to_jd（参数可全空——"
    "服务端会从会话与上传中取）。\n"
    "你可以在需要时依次调用多个工具；得到工具文本后请用中文整理成对用户友好的最终答复。"
)


def _serialize_assistant_message(msg: Any) -> dict[str, Any]:
    """把 SDK 返回的 assistant message 编成可追加到下一轮 messages 的 dict。"""
    """ 这里是把模型调用的tools工具信息进行格式化，供下一轮模型使用"""
    row: dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", None)}
    tc_list = getattr(msg, "tool_calls", None)
    if not tc_list:
        return row
    serial: list[dict[str, Any]] = []
    for tc in tc_list:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") if fn else ""
        args = getattr(fn, "arguments", "") if fn else "{}"
        serial.append(
            {
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", None) or "function",
                "function": {"name": name, "arguments": args or "{}"},
            }
        )
    row["tool_calls"] = serial
    return row


def _dispatch_tool(
    name: str,
    args: dict[str, Any],
    *,
    user_full_text: str,
    session: dict[str, Any],
    llm: Any,
) -> tuple[str, bool]:
    """
    执行单个工具。
    Returns: (发给模型的 tool message 正文, 是否视为「本轮使用过 RAG 检索」)
    """
    from tools import (
        analyze_jd,
        analyze_resume_gap,
        rag_search_tools,
        resolve_jd_plain_for_gap,
        resolve_resume_plain_for_gap,
    )

    try:
        if name == "search_knowledge_base":
            q_model = str(args.get("query") or "").strip()
            if not q_model:
                return "参数错误：`query` 不能为空。", False
            top_k = args.get("top_k")
            k = None
            if isinstance(top_k, int) and top_k > 0:
                k = min(top_k, 32)
            sess_q = ""
            if isinstance(session, dict):
                sess_q = (_build_rag_search_query(user_full_text, session) or "").strip()
            merged = f"{q_model}\n---\n{sess_q}" if sess_q else q_model
            effective = merged[:2000]
            chunks = rag_search_tools(effective, top_k=k)
            if not chunks:
                return (
                    "（知识库未命中：可能尚未建索引，或问题与库内容距离较远。可建议用户执行 `python -m rag_engine` 建索引。）",
                    False,
                )
            ctx = _format_rag_context(chunks)
            return f"【检索到的资料片段】\n{ctx}", True

        if name == "analyze_job_description":
            jd_text = str(args.get("jd_text") or "").strip()
            if len(jd_text) < 40:
                return (
                    "JD 文本过短。请让用户粘贴完整岗位描述到输入框，或将 JD 作为长文本放在消息中。",
                    False,
                )
            return analyze_jd(jd_text, use_llm=True, llm_caller=llm), False

        if name == "compare_resume_to_jd":
            rp = resolve_resume_plain_for_gap(user_full_text, session)
            jp = resolve_jd_plain_for_gap(user_full_text, session)
            return analyze_resume_gap(rp, jp, use_llm=True, llm_caller=llm), False

    except Exception as exc:  # noqa: BLE001
        logger.exception("工具执行失败 %s: %s", name, exc)
        return f"工具执行异常：{exc}", False

    return f"未知工具：{name}", False


def run_tool_agent_turn(
    user_full_text: str,
    session: dict[str, Any],
    llm: Any,
) -> str:
    """
    多轮工具循环；返回最终给用户的中文文本。

    llm 须实现 ``create_chat_completion(messages, tools=..., tool_choice=...)``（见 QwenClient）。
    """
    max_steps = max(1, config.AGENT_TOOL_LOOP_MAX_STEPS)
    used_rag = False
    trace: list[str] = []

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"【用户话轮·含可能的上传标记】\n{user_full_text[:24000]}",
        },
    ]

    for step in range(max_steps):
        try:
            resp = llm.create_chat_completion(
                messages=messages,  # type: ignore[arg-type]
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool Agent 模型调用失败: %s", exc)
            return f"抱歉，带工具编排的本轮对话暂时失败：{exc}。可尝试将环境变量 USE_TOOL_AGENT_LOOP 设为 false 使用传统路由。"

        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if tool_calls:
            trace.extend(
                getattr(getattr(tc, "function", None), "name", "?") or "?" for tc in tool_calls
            )
            messages.append(_serialize_assistant_message(msg))
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                t_name = getattr(fn, "name", "") if fn else ""
                raw_args = getattr(fn, "arguments", "") if fn else "{}"
                tid = getattr(tc, "id", "") or ""
                try:
                    parsed = json.loads(raw_args or "{}")
                    if not isinstance(parsed, dict):
                        parsed = {}
                except json.JSONDecodeError:
                    parsed = {}
                body, did_rag = _dispatch_tool(
                    t_name,
                    parsed,
                    user_full_text=user_full_text,
                    session=session,
                    llm=llm,
                )
                if did_rag:
                    used_rag = True
                if len(body) > 18000:
                    body = body[:18000] + "\n…（输出截断）"
                messages.append(
                    {"role": "tool", "tool_call_id": tid, "content": body}
                )
            continue

        content = (getattr(msg, "content", None) or "").strip()
        if content:
            if isinstance(session, dict):
                session["last_reply_used_rag"] = used_rag
                session["last_agent_tools"] = trace
            return content

        return "抱歉，本轮模型未产出有效文字；请换一种问法或减少工具依赖。"

    if isinstance(session, dict):
        session["last_reply_used_rag"] = used_rag
        session["last_agent_tools"] = trace
    return (
        "抱歉，工具编排达到最大步数仍未结束。**可能原因**：问题需要过多步检索与分析。"
        "请拆成更简单的问题或暂时关闭 USE_TOOL_AGENT_LOOP。"
    )

