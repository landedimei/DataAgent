# -*- coding: utf-8 -*-
"""
app.py — Streamlit 主入口（交互层）

运行：在项目根执行  streamlit run app.py
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import streamlit as st
from openai import APITimeoutError

import config
from agent_brain import QwenClient, run_agent_turn_iter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DE 面试辅导",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 仅用少量样式：不再隐藏 stToolbar，否则侧栏折叠后无法点「展开」箭头
APP_THEME_CSS = """
<style>
    :root {{
        --kb-accent: #1b4332;
        --kb-accent-soft: #d8f3dc;
        --kb-muted: #40916c;
    }}
    .kb-hero {{
        background: linear-gradient(135deg, var(--kb-accent-soft) 0%, #ffffff 55%);
        border: 1px solid #b7e4c7;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }}
    .kb-hero h1 {{
        margin: 0 0 0.35rem 0;
        font-size: 1.45rem;
        color: var(--kb-accent);
        font-weight: 650;
    }}
    footer {{ visibility: hidden; }}
    #MainMenu {{visibility: hidden;}}
</style>
"""
st.markdown(APP_THEME_CSS, unsafe_allow_html=True)


if "messages" not in st.session_state:
    _welcome = """您好，我是**数据开发（DE）方向**的面试与学习助手，可以帮您：

- **知识问答**：数仓、Spark/Flink、SQL、数据质量等；
- **JD 分析**：粘贴岗位描述（可含「分析」「岗位」等字眼）；
- **简历对标**：上传简历 + 附带 JD 或先分析 JD，再问「差距」「匹配」等；
- **模拟面试**：说一句「**开始模拟面试**」，按阶段出题，结束时说「**请总结**」或「**结束面试**」。

需要把教材、面经放进知识库时，请用**左侧栏「资料入库」**；入库后可被对话里的检索引用。

---
有什么想聊的，直接在下框输入即可。"""
    st.session_state["messages"] = [{"role": "assistant", "content": _welcome}]


# ---------------------------------------------------------------------------
# API Key
# ---------------------------------------------------------------------------
if not config.is_api_configured():
    st.error("请在项目根目录 `.env` 中配置有效的 `DASHSCOPE_API_KEY`。")
    st.stop()
    sys.exit(1)

if "qwen" not in st.session_state:
    try:
        st.session_state["qwen"] = QwenClient()
    except Exception as exc:  # noqa: BLE001
        logger.exception("初始化客户端失败: %s", exc)
        st.error("无法连接模型，请检查 .env 与网络。")
        st.stop()
        sys.exit(1)

llm: QwenClient = st.session_state["qwen"]


def _maybe_append_upload_text(user_prompt: str, state: Any) -> str:
    name = state.get("last_upload_name")
    b = state.get("last_upload_bytes")
    if not name or b is None:
        return user_prompt
    from tools import extract_plain_from_upload

    extra = extract_plain_from_upload(name, b)
    return user_prompt + "\n\n[上传文件内容]\n" + extra[:15000]


def _render_kb_ingest_sidebar() -> None:
    try:
        from rag_engine import ingest_uploaded_bytes
    except ImportError as exc:
        st.caption(f"知识库不可用：{exc}")
        return

    kb_file = st.file_uploader(
        "选择文件",
        type=["pdf", "md", "txt", "markdown", "docx"],
        label_visibility="collapsed",
        key="sb_kb_file",
    )
    kb_kind = st.segmented_control(
        "类型",
        options=["面试面经", "学习资料"],
        default="面试面经",
        key="sb_kb_kind",
        label_visibility="collapsed",
    )

    if not st.button("上传并入库", type="primary", use_container_width=True, key="sb_kb_run"):
        return

    if kb_file is None:
        st.warning("请先选择文件。")
        return

    kind = kb_kind if kb_kind in ("面试面经", "学习资料") else "面试面经"
    cat_key = "interview" if kind == "面试面经" else "learning"
    raw = kb_file.getvalue()

    try:
        with st.spinner("入库中…"):
            result = ingest_uploaded_bytes(raw, kb_file.name, category_key=cat_key)
        if result.get("ok"):
            rel = result.get("relative", "")
            nc = result.get("chunks")
            tail = f"（{nc} 块）" if isinstance(nc, int) else ""
            st.success(f"已入库：`{rel}` {tail}".strip())
        else:
            st.error(str(result.get("error", result)))
    except APITimeoutError:
        st.error(
            "向量服务请求超时：`dashscope.aliyuncs.com` 无法在设定时间内连通（常见于网络不稳定、"
            "公司防火墙或需代理）。可在 `.env` 中增大 `EMBEDDING_HTTP_TIMEOUT_SECONDS`（如 `300`），"
            "并检查本机是否能访问阿里云 DashScope。"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("入库失败: %s", exc)
        st.error(str(exc))


# ---------------------------------------------------------------------------
# 侧栏：入库（箭头收起后仍可点页面左上角「>`」展开；请勿再隐藏 Toolbar）
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("###### 资料入库")
    st.caption("落盘至 data/ 并写入向量库，便于对话检索。")
    _render_kb_ingest_sidebar()


# ---------------------------------------------------------------------------
# 主区
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="kb-hero"><h1>数据开发 · 面试辅导</h1>'
    '<p style="margin:0;color:#40916c;font-size:0.95rem;">'
    "知识问答 · JD · 简历对标 · 模拟面试</p></div>",
    unsafe_allow_html=True,
)


for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("输入消息…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                run_agent_turn_iter(
                    user_text=_maybe_append_upload_text(prompt, st.session_state),
                    session=st.session_state,  # type: ignore[arg-type]
                    llm=llm,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("对话异常: %s", exc)
            reply = "抱歉，暂时不可用，请稍后再试。"
            st.markdown(reply)

    st.session_state["messages"].append({"role": "assistant", "content": reply})
