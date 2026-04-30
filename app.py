# -*- coding: utf-8 -*-
"""
app.py — Streamlit 主入口（交互层）

运行方式（在项目根目录、已激活虚拟环境的前提下）:
    streamlit run app.py

数据流简图:
用户打开浏览器 → 本脚本渲染侧栏+聊天区 → 用户输入/上传
→ 若 API Key 合法则把文本交给 agent_brain.run_agent_turn
→ 将助手回复拼进 st.session_state["messages"] 再 re-run 显示。
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import streamlit as st

import config
from agent_brain import QwenClient, run_agent_turn

# 基础日志，便于在终端看到 API 侧异常（不暴露 Key）
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 页面与 session_state 初始化
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DE 智能面试辅导 Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "你好，我是数据开发（DE）方向的面试辅导助手。可以问我知识点、"
            "粘贴 JD 做分析，或说「开始模拟面试」。首次使用请先在左侧配置 API Key。",
        }
    ]


# ---------------------------------------------------------------------------
# 启动时 API Key 校验：未配置则阻止继续使用（功能验收第 1 条）
# ---------------------------------------------------------------------------
if not config.is_api_configured():
    st.error(
        "未检测到有效的 DASHSCOPE_API_KEY。请在项目根目录创建 `.env` 文件，"
        "并写入: `DASHSCOPE_API_KEY=你的通义Key`（可参考 `.env.example`）。\n"
        "出于安全，不要把 Key 写进代码或提交到 Git。"
    )
    st.stop()
    # 非 `streamlit run` 时 st.stop 可能不抛异常，避免继续执行导致未初始化 session
    sys.exit(1)

# 能执行到这里，说明可以安全构造客户端（在 Sidebar 中也可展示“已配置”状态）
# 为减少重复建连，把 client 放 session
if "qwen" not in st.session_state:
    try:
        st.session_state["qwen"] = QwenClient()
    except Exception as exc:  # noqa: BLE001
        logger.exception("初始化 QwenClient 失败: %s", exc)
        st.error("无法初始化大模型客户端，请检查 .env 与网络。将阻止继续操作。")
        st.stop()
        sys.exit(1)

llm: QwenClient = st.session_state["qwen"]


def _maybe_append_upload_text(user_prompt: str, state: Any) -> str:
    """
    若用户刚上传了文件，把其解析文本拼在用户问题后，让下游工具/模型「看得见」内容。

    使用单独函数、并放在 `chat_input` 之前，避免「先调用后定义」在 Streamlit
    重跑时触发 NameError。
    """
    name = state.get("last_upload_name")
    b = state.get("last_upload_bytes")
    if not name or b is None:
        return user_prompt
    from tools import extract_plain_from_upload

    extra = extract_plain_from_upload(name, b)
    return user_prompt + "\n\n[上传文件内容]\n" + extra[:15000]


# ---------------------------------------------------------------------------
# 侧栏：配置说明与（可选）文件上传
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("设置与说明")
    st.success("API Key 已从环境变量加载（本界面不显示明文）。")
    st.caption(
        f"Base URL: `{config.QWEN_BASE_URL}`  |  模型: `{config.QWEN_CHAT_MODEL}`\n\n"
        f"多轮工具编排（Agent）：**{'已开启（USE_TOOL_AGENT_LOOP）' if config.USE_TOOL_AGENT_LOOP else '关闭（默认仍为「路由+规则」）'}**。"
    )
    st.divider()
    st.subheader("向量知识库")
    try:
        from rag_engine import (
            collection_chunk_count,
            incremental_sync_from_disk,
            index_knowledge_base_full_rebuild,
            ingest_uploaded_bytes,
            project_data_dir,
        )

        kb_count = collection_chunk_count()
        st.caption(
            f"当前 **{kb_count}** 条向量  ·  文件放在 `data/{config.KB_DIR_INTERVIEW}/` 或 `data/{config.KB_DIR_LEARNING}/`"
        )
        kb_file = st.file_uploader(
            "上传教材 / 面经（落盘 + 向量化）",
            type=["pdf", "md", "txt", "markdown", "docx"],
            key="kb_vector_ingest",
            help="需 API Key 做 Embedding。文件会保存到 data/ 对应子目录。",
        )
        kb_kind = st.radio(
            "归入类型",
            ("面试面经", "学习资料"),
            horizontal=True,
            key="kb_kind_radio",
        )
        if st.button("解析并写入向量库", type="primary", key="kb_ingest_run"):
            if kb_file is None:
                st.warning("请先选择一个文件。")
            else:
                cat_key = "interview" if kb_kind == "面试面经" else "learning"
                raw = kb_file.getvalue()
                with st.spinner("正在解析、切块、写入 Chroma …"):
                    try:
                        result = ingest_uploaded_bytes(raw, kb_file.name, category_key=cat_key)
                        if result.get("ok"):
                            st.success(f"已保存并索引：`{result.get('relative')}`")
                            if result.get("sync_detail"):
                                st.json(result["sync_detail"])
                        else:
                            st.error(str(result))
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("知识库入库失败: %s", exc)
                        st.error(f"失败：{exc}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("增量同步整个 data", key="incr_sync_kb"):
                with st.spinner("比对 manifest 与文件哈希…"):
                    try:
                        st.session_state["_last_kb_sync"] = incremental_sync_from_disk()
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("增量同步异常: %s", exc)
                        st.error(str(exc))
        with c2:
            purge = st.checkbox("确认清空后全量重建", key="kb_confirm_rebuild_cb")
            if st.button("全量重建向量库", key="full_rebuild_kb", disabled=not purge):
                with st.spinner("清空集合并重新嵌入…"):
                    try:
                        r = index_knowledge_base_full_rebuild(clear_before=True)
                        st.session_state["_last_kb_sync"] = r
                        st.success("全量重建完成。")
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("全量重建失败: %s", exc)
                        st.error(str(exc))
        last_sync = st.session_state.get("_last_kb_sync")
        if isinstance(last_sync, dict):
            st.caption("最近一次同步结果：")
            st.json(last_sync)
        st.caption(f"`data` 根目录：`{project_data_dir()}`")
    except ImportError as exc:
        st.info(f"向量库模块不可用：{exc}")

    st.divider()
    st.subheader("资料上传（实验性）")
    st.caption(
        "按需求，上传内容默认不持久化；仅当前会话在内存中解析。"
        f" 调试落盘开关: {config.PERSIST_UPLOADS_TO_DISK}"
    )
    up = st.file_uploader(
        "简历或 JD 文件（PDF / MD / TXT）",
        type=["pdf", "md", "txt", "markdown"],
        key="resume_side_upload",
    )
    if up is not None:
        raw = up.getvalue()
        st.session_state["last_upload_name"] = up.name
        st.session_state["last_upload_bytes"] = raw
        st.info(f"已接收文件 `{up.name}`，大小 {len(raw)} 字节。发送消息时可在逻辑里使用。")
    st.divider()
    st.markdown(
        "操作提示：\n"
        "- 粘贴 **JD** 并含「分析/岗位」等词 → JD 分析。\n"
        "- **模拟面试（建议先上传简历 PDF/MD）**：流程为——项目深挖 → 八股与场景 → 开放题 → **SQL** → **算法**。\n"
        "说「请总结 / 结束面试」生成综合评价。\n"
        "- 知识库：`data/` 建索引后与面经检索配合。"
    )

# ---------------------------------------------------------------------------
# 主区：聊天气泡
# ---------------------------------------------------------------------------
st.title("数据开发（DE）智能面试辅导")
st.caption("纯 Python + OpenAI 兼容 + Streamlit 教学 Demo 框架")

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------------------------------------------------------------------
# 输入框：多轮 session_history → Agent → 写回
# ---------------------------------------------------------------------------
if prompt := st.chat_input("输入你的问题、粘贴 JD，或说「开始模拟面试」…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中（含意图路由/可选工具）…"):
            # session_state 是 dict-like，直接传给 run_agent_turn 以更新 last_intent 等
            try:
                reply = run_agent_turn(
                    user_text=_maybe_append_upload_text(prompt, st.session_state),
                    session=st.session_state,  # type: ignore[arg-type]
                    llm=llm,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("主循环异常: %s", exc)
                reply = "抱歉，服务暂时不可用，请稍后再试。"

        st.markdown(reply)
        if st.session_state.get("mock_interview_active"):
            st.caption("模拟面试进行中；输入「请总结」或「结束面试」可结束并生成评价。")
        # agent_brain 在命中 RAG 时写入 last_reply_used_rag，供界面提示
        elif st.session_state.get("last_reply_used_rag"):
            st.caption("本轮已结合「本地知识库 RAG」检索到的片段作答。")

    st.session_state["messages"].append({"role": "assistant", "content": reply})
