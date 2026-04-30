# -*- coding: utf-8 -*-
"""
与 agent_brain 中 RAG 相关的单元测试（不调用真实 DashScope / 不写真实 Chroma）。

使用 unittest.mock 隔离「检索是否为空」「是否返回块」「LLM 是否被调用」。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent_brain import (
    UserIntent,
    _answer_with_rag_grounding,
    _build_rag_search_query,
    _format_rag_context,
    _intent_allows_rag,
    _maybe_boost_mock_search_query,
    _rag_system_template_for_intent,
)


def test_intent_allows_rag_covers_mock_and_knowledge():
    assert _intent_allows_rag(UserIntent.KNOWLEDGE) is True
    assert _intent_allows_rag(UserIntent.GENERAL) is True
    assert _intent_allows_rag(UserIntent.MOCK_INTERVIEW) is True
    assert _intent_allows_rag(UserIntent.JD_ANALYSIS) is False
    assert _intent_allows_rag(UserIntent.RESUME_UPLOAD) is False


def test_format_rag_context_numbered_blocks():
    ctx = _format_rag_context(["第一段", "第二段"])
    assert "### 片段 1" in ctx
    assert "### 片段 2" in ctx
    assert "第一段" in ctx and "第二段" in ctx


def test_build_rag_search_query_with_prev_session():
    session: dict = {"last_user_message": "什么是 Hive？"}
    q = _build_rag_search_query("和 Spark 有啥区别？", session)
    assert "Hive" in q
    assert "Spark" in q
    assert "---" in q


def test_build_rag_search_query_no_prev():
    assert _build_rag_search_query("仅一句", {}) == "仅一句"


def test_maybe_boost_mock_appends_hint():
    out = _maybe_boost_mock_search_query("开始模拟面试", UserIntent.MOCK_INTERVIEW)
    assert "面经" in out
    assert "开始模拟面试" in out


def test_maybe_boost_mock_empty_query_uses_hint_only():
    out = _maybe_boost_mock_search_query("", UserIntent.MOCK_INTERVIEW)
    assert "面经" in out


def test_maybe_boost_no_op_for_knowledge():
    s = "仅知识点"
    assert _maybe_boost_mock_search_query(s, UserIntent.KNOWLEDGE) == s


def test_rag_system_template_for_intent():
    t_mock = _rag_system_template_for_intent(UserIntent.MOCK_INTERVIEW)
    t_k = _rag_system_template_for_intent(UserIntent.KNOWLEDGE)
    assert "面试官" in t_mock
    assert "{context}" in t_mock
    assert "面试辅导助手" in t_k or "助手" in t_k
    assert t_mock != t_k


@patch("agent_brain.is_vector_store_empty", return_value=True)
@patch("agent_brain.rag_search")
def test_answer_with_rag_grounding_skips_when_store_empty(mock_search, _mock_empty):
    """向量库空时不应调 rag_search。"""
    llm = MagicMock()
    out = _answer_with_rag_grounding(
        "问", llm, "q", used_rag=[], intent=UserIntent.KNOWLEDGE
    )
    assert out is None
    mock_search.assert_not_called()
    llm.simple_chat.assert_not_called()


@patch("agent_brain.is_vector_store_empty", return_value=False)
@patch("agent_brain.rag_search", return_value=[])
def test_answer_with_rag_grounding_no_chunks_no_llm(_mock_rag, _mock_empty):
    llm = MagicMock()
    out = _answer_with_rag_grounding(
        "问", llm, "q", used_rag=[], intent=UserIntent.GENERAL
    )
    assert out is None
    llm.simple_chat.assert_not_called()


@patch("agent_brain.is_vector_store_empty", return_value=False)
@patch("agent_brain.rag_search", return_value=["面经里考了 Spark 宽依赖"])
def test_answer_with_rag_grounding_mock_uses_interviewer_system(_mock_rag, _mock_empty):
    llm = MagicMock()
    llm.simple_chat = MagicMock(return_value="好，我出题了")
    used: list[bool] = []
    out = _answer_with_rag_grounding(
        "来一道题",
        llm,
        "面经 面试",
        used_rag=used,
        intent=UserIntent.MOCK_INTERVIEW,
    )
    assert out == "好，我出题了"
    assert used == [True]
    llm.simple_chat.assert_called_once()
    system_arg, user_arg = llm.simple_chat.call_args[0]
    assert "面试官" in system_arg
    assert "面经" in system_arg or "参考" in system_arg
    assert "面试进行中" in user_arg


@patch("agent_brain.is_vector_store_empty", return_value=False)
@patch("agent_brain.rag_search", return_value=["知识点A"])
def test_answer_with_rag_grounding_knowledge_user_block(_mock_rag, _mock_empty):
    llm = MagicMock()
    llm.simple_chat = MagicMock(return_value="答")
    used: list[bool] = []
    _answer_with_rag_grounding(
        "问什么", llm, "q", used_rag=used, intent=UserIntent.KNOWLEDGE
    )
    _, user_arg = llm.simple_chat.call_args[0]
    assert "用户当前问题" in user_arg
    assert "面试进行中" not in user_arg
