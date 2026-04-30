# -*- coding: utf-8 -*-
"""模拟面试状态机与终局评价：mock LLM / 不连网。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent_brain import (
    UserIntent,
    _build_mock_stage_instruction_block,
    _format_completed_mock_rounds,
    _format_mock_transcript_for_eval,
    _is_mock_end_request,
    _keyword_wants_mock_start,
    _mock_final_evaluation_and_reset,
    _mock_start_interview,
    _try_mock_interview_state_machine,
)


def test_is_mock_end_request_short_phrases():
    assert _is_mock_end_request("结束面试") is True
    assert _is_mock_end_request("请总结") is True
    assert _is_mock_end_request("我觉得宽依赖会触发 shuffle " * 20) is False


def test_keyword_wants_mock_start():
    assert _keyword_wants_mock_start("开始模拟面试") is True
    assert _keyword_wants_mock_start("讲一讲 hive 分区" * 10) is False


def test_format_completed_rounds():
    s = {
        "mock_transcript": [
            {"question": "Q1", "answer": "A1"},
        ],
    }
    t = _format_completed_mock_rounds(s)
    assert "Q1" in t and "A1" in t


def test_format_eval_includes_unanswered_last():
    s = {
        "mock_transcript": [{"question": "Q1", "answer": "A1"}],
        "mock_last_question": "这是还没答的第二题长文本",
    }
    t = _format_mock_transcript_for_eval(s)
    assert "Q1" in t
    assert "未作答" in t
    assert "第二题" in t


@patch("agent_brain._summarize_resume_for_mock", return_value="【摘要】在某厂做离线数仓")
@patch("agent_brain._load_resume_plain_from_session", return_value="工作经历：大厂 数据仓库")
@patch("agent_brain._run_mock_rag_or_llm", return_value="第一题：说说 Spark 宽依赖？")
def test_start_interview_sets_session(_rr: MagicMock, _ldr: MagicMock, _sr: MagicMock) -> None:
    s: dict = {}
    llm = MagicMock()
    out = _mock_start_interview("开始模拟面试", s, llm)
    assert out == "第一题：说说 Spark 宽依赖？"
    assert s.get("mock_interview_active") is True
    assert s.get("mock_awaiting_answer") is True
    assert s.get("mock_last_question") == out
    assert s.get("mock_emit_index") == 1
    assert "mock_resume_plain" in s


def test_project_stage_instruction_mentions_resume():
    blk = _build_mock_stage_instruction_block(
        "project_deep",
        resume_digest="某项目用的是 Spark SQL",
        resume_excerpt="项目A",
        history="（无）",
    )
    assert "项目" in blk or "实习" in blk
    assert "难点" in blk or "深挖" in blk


def test_sql_stage_contains_sql_keyword():
    blk = _build_mock_stage_instruction_block(
        "sql_coding", resume_digest="x", resume_excerpt="y", history="（无）"
    )
    assert "SQL" in blk


def test_algo_stage_algo_keyword():
    blk = _build_mock_stage_instruction_block(
        "algo_coding", resume_digest="x", resume_excerpt="y", history="（无）"
    )
    assert "算法" in blk or "LeetCode" in blk


def test_final_eval_resets_session():
    s = {
        "mock_interview_active": True,
        "mock_awaiting_answer": True,
        "mock_transcript": [{"question": "Q1", "answer": "A1"}],
        "mock_last_question": "",
    }
    llm = MagicMock()
    llm.simple_chat = MagicMock(return_value="## 整体印象\n不错")
    r = _mock_final_evaluation_and_reset("结束面试", s, llm)  # type: ignore[arg-type]
    assert "整体" in r
    assert s.get("mock_interview_active") is None
    assert s.get("mock_transcript") is None
    llm.simple_chat.assert_called_once()


@patch("agent_brain._mock_start_interview", return_value="Q1")
def test_state_machine_starts_on_intent(_ms: MagicMock) -> None:
    s: dict = {}
    llm = MagicMock()
    r = _try_mock_interview_state_machine("开始", s, llm, UserIntent.MOCK_INTERVIEW)
    assert r == "Q1"
    _ms.assert_called_once()


@patch("agent_brain._mock_final_evaluation_and_reset", return_value="报告")
def test_state_machine_end_when_active(mfe: MagicMock) -> None:
    s = {"mock_interview_active": True, "mock_awaiting_answer": True}
    llm = MagicMock()
    r = _try_mock_interview_state_machine("请总结", s, llm, UserIntent.GENERAL)
    assert r == "报告"
    mfe.assert_called_once()
