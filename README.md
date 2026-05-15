# DataAgent — DE 智能面试辅导 Agent

> 一个面向**数据开发（Data Engineering）**岗位的 AI 面试辅导助手，帮助求职者准备技术面试、分析 JD、做简历 Gap 诊断，并支持完整的模拟面试流程。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

---

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 🧠 **意图识别** | LLM 零样本分类 + 关键词规则兜底，准确路由用户意图 |
| 📋 **JD 深度分析** | 粘贴岗位描述，自动提取核心技术栈与高频考点 |
| 📚 **RAG 知识库问答** | 混合检索（BM25 + 向量语义）+ RRF 重排序，精准回答 DE 技术问题 |
| 📄 **简历 Gap 分析** | 对比简历与 JD，指出技术短板与学习建议 |
| 🎯 **模拟面试** | 完整五阶段面试（项目深挖 → 八股 → 开放题 → SQL → 算法），面试结束后生成综合评价报告 |
| 📂 **知识库管理** | 支持增量同步、全量重建，可上传 PDF / MD / DOCX 入库 |

---

## 🏗️ 架构设计

```
用户输入
   │
   ▼
┌─────────────────────────────────────────────┐
│  交互层 (Streamlit UI)      app.py           │
│  聊天窗口 / 文件上传 / 侧边栏知识库管理        │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│  核心控制层 (Agent Brain)   agent_brain.py   │
│  意图路由 → ReAct 推理 → 工具调度             │
└──────┬───────────────┬──────────────────────┘
       │               │
       ▼               ▼
┌──────────┐   ┌───────────────────────────────┐
│ 工具层    │   │  RAG 检索层    rag_engine.py   │
│ tools.py │   │  BM25 + 向量混合检索 + RRF     │
│ JD分析   │   │  ChromaDB 本地持久化           │
│ Gap分析  │   │  通义千问 Embedding API        │
└──────────┘   └───────────────────────────────┘
```

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| Web UI | Streamlit |
| LLM | 通义千问（OpenAI 兼容模式 / DashScope） |
| 向量库 | ChromaDB（本地持久化） |
| 混合检索 | rank_bm25 + jieba 中文分词 + RRF 重排序 |
| 文档解析 | pdfplumber（PDF）/ python-docx（DOCX）/ 原生（MD / TXT） |
| HTTP 重试 | tenacity（指数退避） |
| 测试 | pytest |

---

## 📁 项目结构

```
DataAgent/
├── app.py              # Streamlit 主程序入口（交互层）
├── agent_brain.py      # Agent 核心逻辑（意图识别、ReAct 循环、模拟面试状态机）
├── tool_agent.py       # OpenAI 工具调用多轮循环（可选）
├── tools.py            # 工具函数（JD 分析、简历 Gap 分析、文档解析）
├── rag_engine.py       # RAG 引擎（混合检索、向量化、Chroma 管理）
├── config.py           # 配置管理（API Key、超参数）
├── requirements.txt    # 依赖列表
├── .env                # 环境变量（不提交到 Git）
├── .gitignore
├── data/
│   ├── 数据开发面经/   # 面试面经文档（PDF / MD / DOCX）
│   └── 学习资料/       # 学习资料文档
├── vector_db/          # ChromaDB 持久化目录（自动生成，不提交）
└── tests/              # 单元测试
    ├── test_rag_engine.py
    ├── test_agent_rag.py
    └── test_mock_interview.py
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone git@github.com:landedimei/DataAgent.git
cd DataAgent
```

### 2. 创建并激活虚拟环境

```bash
python3 -m venv DataAgent-env
source DataAgent-env/bin/activate   # Windows: DataAgent-env\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 API Key

复制示例文件并填入你的通义千问（DashScope）API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
DASHSCOPE_API_KEY=sk-你的真实密钥
```

> 在[阿里云 DashScope 控制台](https://dashscope.aliyun.com/)申请 API Key。

### 5. 建立知识库索引（可选，有资料时）

把 PDF / MD / DOCX 文件放入 `data/学习资料/` 或 `data/数据开发面经/`，然后执行：

```bash
# 全量重建（首次）
python -m rag_engine rebuild

# 增量同步（后续新增文件）
python -m rag_engine incremental
```

### 6. 启动应用

```bash
streamlit run app.py
```

打开浏览器访问 `http://localhost:8501`。

---

## 💬 使用指南

| 你想做什么 | 怎么说 / 操作 |
|------------|---------------|
| 提问 DE 知识 | 直接输入，如「Spark 宽依赖和窄依赖的区别？」 |
| 分析岗位 JD | 把 JD 文本粘贴到对话框 |
| 简历 Gap 分析 | 侧边栏上传简历 PDF + 对话框粘贴 JD，发送「帮我做 Gap 分析」 |
| 开始模拟面试 | 发送「开始模拟面试」（建议先上传简历） |
| 结束模拟面试 | 发送「请总结」或「结束面试」，生成综合评价报告 |
| 上传资料到知识库 | 侧边栏「上传教材 / 面经」区域 |

---

## ⚙️ 常用配置（`.env`）

```env
# 必填
DASHSCOPE_API_KEY=sk-xxx

# 可选：自定义模型
QWEN_CHAT_MODEL=qwen-plus          # 聊天模型（qwen-turbo / qwen-plus / qwen-max）
QWEN_EMBEDDING_MODEL=text-embedding-v3

# 可选：RAG 检索参数
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=10
RAG_HYBRID_ALPHA=0.6               # 向量权重（1-alpha 为 BM25 权重）
RAG_SIMILARITY_THRESHOLD=0.3       # 向量相似度过滤阈值

# 可选：启用 OpenAI 工具调用多轮循环
USE_TOOL_AGENT_LOOP=false
```

---

## 🧪 运行测试

```bash
pytest tests/ -v
```

---

## 📖 核心模块说明

### RAG 混合检索

本项目实现了 **BM25 关键词检索 + 向量语义检索 + RRF 重排序** 的混合策略：

- **BM25**：精准匹配关键词（如搜「Flink checkpoint」能精准命中包含这两个词的段落）
- **向量检索**：语义理解（如搜「实时计算容错」能匹配到讲 Flink checkpoint 的内容）
- **RRF（Reciprocal Rank Fusion）**：将两种结果按排名加权融合，通常优于任一单一方法
- **动态 Top-K**：根据查询复杂度自动调整返回条数（3 ~ 15 条）

### 模拟面试五阶段流程

```
开始面试
  │
  ├─ 1. 项目 / 实习深挖（根据简历追问）
  ├─ 2. 技术八股 + 场景题（数仓 / Spark / Flink / SQL 优化等）
  ├─ 3. 开放性问题（技术视野、职业规划）
  ├─ 4. SQL 手写题（给表结构和业务需求）
  └─ 5. 算法题（LeetCode 中等偏易）
         │
         └─ 说「请总结」→ 生成综合评价报告
```

---

## 🔒 安全说明

- API Key 存储在 `.env` 文件中，**不提交到版本控制**
- 上传的简历 / JD **默认不落盘**（仅在当前会话内存中解析）
- 如需调试落盘，在 `.env` 中设置 `PERSIST_UPLOADS_TO_DISK=true`

---

## 📄 License

[Apache License 2.0](LICENSE)
