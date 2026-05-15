# -*- coding: utf-8 -*-
"""
config.py — 集中管理配置与超参数

设计思路（为什么需要单独一个模块）：
- 将「从环境读 Key」「默认模型名」「RAG 分块大小」等集中在一处，避免魔法数字散落在多文件里。
- 改一处，全局生效，便于你后续做实验（例如把 chunk_size 从 500 改成 300）。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 在「导入 config 的第一次」时，从项目根目录加载 .env
# 为什么用 Path：不依赖「当前工作目录」一定是项目根，尽量按本文件位置推算根目录
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# API / 通义千问（OpenAI 兼容模式）
# ---------------------------------------------------------------------------
# DashScope 官方 OpenAI 兼容基地址；chat / embeddings 共用此前缀（v1）。
# 注意：v3 与 OpenAI SDK 拼出的 /embeddings 路径不匹配，会得到 404。
QWEN_BASE_URL: str = os.getenv(
    "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# 聊天模型名：可按控制台可用列表更换，例如 qwen-turbo、qwen-plus
QWEN_CHAT_MODEL: str = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")
# 向量模型名：用于 RAG 向量化时若走「API Embedding」
QWEN_EMBEDDING_MODEL: str = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

# 为什么用 DASHSCOPE_API_KEY 命名：与阿里云官方环境变量名一致，减少歧义
DASHSCOPE_API_KEY: str | None = os.getenv("DASHSCOPE_API_KEY")
# 兼容部分文档写作习惯：也允许用 OPENAI_API_KEY 存同一个 Key
if not DASHSCOPE_API_KEY:
    DASHSCOPE_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# HTTP 调用（需求：超时 60s、最多 3 次、指数退避由调用侧 tenacity 实现）
# ---------------------------------------------------------------------------
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
# Embedding 常为批量调用，网络抖动时易出现 ConnectTimeout，可单独调大超时与重试次数
EMBEDDING_HTTP_TIMEOUT_SECONDS: float = float(os.getenv("EMBEDDING_HTTP_TIMEOUT_SECONDS", "180"))
EMBEDDING_MAX_RETRIES: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "4"))

# ---------------------------------------------------------------------------
# RAG 分块与检索
# ---------------------------------------------------------------------------
RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "10"))
# 混合检索：BM25 与向量检索的权重。向量权重越大，语义越重要；BM25 越大，关键词匹配越重要。
RAG_HYBRID_ALPHA: float = float(os.getenv("RAG_HYBRID_ALPHA", "0.6"))
# 检索结果相似度阈值：低于此分数的向量检索结果会被过滤掉，避免返回不相关的内容
RAG_SIMILARITY_THRESHOLD: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))
# 动态 Top-K：根据查询复杂度调整返回条数的范围
RAG_TOP_K_MIN: int = int(os.getenv("RAG_TOP_K_MIN", "3"))
RAG_TOP_K_MAX: int = int(os.getenv("RAG_TOP_K_MAX", "15"))

# Chroma 持久化目录：相对项目根，便于和 .gitignore 里 vector_db/ 一致
CHROMA_PERSIST_DIR: Path = _PROJECT_ROOT / "vector_db"

# data/ 下知识库子目录（与本地文件夹名一致，可 .env 覆盖）
KB_DIR_INTERVIEW: str = os.getenv("KB_DIR_INTERVIEW", "数据开发面经")
KB_DIR_LEARNING: str = os.getenv("KB_DIR_LEARNING", "学习资料")

# ---------------------------------------------------------------------------
# 调试：若将来需要把上传文件落盘，必须显式开 True，并在界面上说明隐私风险
# ---------------------------------------------------------------------------
PERSIST_UPLOADS_TO_DISK: bool = os.getenv("PERSIST_UPLOADS_TO_DISK", "false").lower() in (
    "1",
    "true",
    "yes",
)

# ---------------------------------------------------------------------------
# Agent：OpenAI 兼容「工具调用」多轮循环（见 tool_agent.run_tool_agent_turn）
# ---------------------------------------------------------------------------
USE_TOOL_AGENT_LOOP: bool = os.getenv("USE_TOOL_AGENT_LOOP", "false").lower() in (
    "1",
    "true",
    "yes",
)
AGENT_TOOL_LOOP_MAX_STEPS: int = int(os.getenv("AGENT_TOOL_LOOP_MAX_STEPS", "8"))


def is_api_configured() -> bool:
    """
    判断 API Key 是否已配置（非空、且不是明显占位符）。

    为什么不仅判断「有环境变量」：
    学生常复制 .env.example 却忘记改 your_api_key_here，这种应在启动时就被拦住。
    """
    if not DASHSCOPE_API_KEY:
        return False
    key = DASHSCOPE_API_KEY.strip()
    if key.lower() in ("your_api_key_here", "sk-placeholder", "replace_me"):
        return False
    return len(key) >= 8
