# -*- coding: utf-8 -*-
"""
测试文件：覆盖 rag_engine 的核心功能
- 分块逻辑
- BM25 关键词检索
- 向量检索（mock）
- RRF 混合重排序
- 动态 Top-K
- 多轮对话检索增强
- 中文分词
- 索引健康检查
"""
from unittest.mock import MagicMock, patch
from rag_engine import (
    simple_chunk_text,
    tokenize_chinese,
    bm25_search,
    vector_search_with_scores,
    rrf_fusion,
    compute_dynamic_top_k,
    build_multi_turn_search_context,
    _load_all_documents_for_bm25,
    check_index_health,
    get_index_stats,
)


# ---------------------------------------------------------------------------
# 1. 分块逻辑（保留原有测试）
# ---------------------------------------------------------------------------
def test_chunk_overlap():
    text = "a" * 100
    parts = simple_chunk_text(text, chunk_size=30, overlap=5)
    assert len(parts) >= 2
    assert parts[0][-5:] == parts[1][:5]


def test_chunk_empty_text():
    assert simple_chunk_text("") == []
    assert simple_chunk_text("   ") == []


def test_chunk_shorter_than_size():
    text = "hello"
    parts = simple_chunk_text(text, chunk_size=100, overlap=10)
    assert parts == ["hello"]


# ---------------------------------------------------------------------------
# 2. 中文分词
# ---------------------------------------------------------------------------
def test_tokenize_chinese():
    tokens = tokenize_chinese("数据仓库中ODS层的设计原则")
    assert len(tokens) > 0
    # 分词后应该包含有意义的词（不是单个字符）
    joined = "".join(tokens)
    assert "数据仓库" in joined or "数据" in joined


def test_tokenize_empty():
    assert tokenize_chinese("") == []
    assert tokenize_chinese("   ") == []


def test_tokenize_punctuation_filtered():
    tokens = tokenize_chinese("你好，世界！")
    # 标点符号"，"和"！"应该被过滤（单字符）
    assert "，" not in tokens
    assert "！" not in tokens


# ---------------------------------------------------------------------------
# 3. BM25 关键词检索
# ---------------------------------------------------------------------------
def test_bm25_basic():
    docs = [
        "Spark 的 shuffle 机制是大数据处理的核心概念",
        "Hive 数仓分层设计中 ODS 层用于存放原始数据",
        "Flink 的 checkpoint 机制保证了实时计算的容错能力",
        "SQL 优化中的索引设计可以大幅提升查询性能",
    ]
    ids = ["doc_0", "doc_1", "doc_2", "doc_3"]
    # 搜索含"shuffle"的文档
    results = bm25_search("Spark shuffle", docs, ids, top_k=2)
    assert len(results) > 0
    # 第一个结果应该包含 "shuffle"
    assert "shuffle" in results[0][1]


def test_bm25_no_match():
    docs = ["数据仓库分层设计", "ETL 流程规范"]
    ids = ["d1", "d2"]
    results = bm25_search("量子计算", docs, ids, top_k=5)
    # 可能返回空或分数极低，但不应报错
    assert isinstance(results, list)


def test_bm25_empty_inputs():
    assert bm25_search("query", [], [], top_k=5) == []
    assert bm25_search("", ["doc"], ["id"], top_k=5) == []
    assert bm25_search("   ", ["doc"], ["id"], top_k=5) == []


def test_bm25_top_k_limit():
    docs = [f"文档编号{i}包含关键词test" for i in range(20)]
    ids = [f"d{i}" for i in range(20)]
    results = bm25_search("test", docs, ids, top_k=5)
    assert len(results) <= 5


def test_bm25_chinese_search():
    docs = [
        "宽依赖和窄依赖是Spark RDD的核心概念",
        "Flink通过两阶段提交协议实现exactly-once语义",
        "Kafka的消费者组机制实现了负载均衡",
        "宽依赖会导致shuffle操作，影响任务性能",
    ]
    ids = ["d0", "d1", "d2", "d3"]
    results = bm25_search("宽依赖 shuffle", docs, ids, top_k=3)
    assert len(results) > 0
    # 应该命中包含"宽依赖"和"shuffle"的文档
    top_ids = [r[0] for r in results]
    assert "d0" in top_ids or "d3" in top_ids


# ---------------------------------------------------------------------------
# 4. RRF 混合重排序
# ---------------------------------------------------------------------------
def test_rrf_fusion_basic():
    bm25_results = [("d1", "doc1", 5.0), ("d2", "doc2", 3.0), ("d3", "doc3", 1.0)]
    vector_results = [("d2", "doc2", 0.9), ("d1", "doc1", 0.8), ("d4", "doc4", 0.7)]

    fused = rrf_fusion(bm25_results, vector_results)
    assert len(fused) > 0

    # d1 和 d2 同时被两种方法命中，应该排在前面
    top_ids = [f[0] for f in fused[:2]]
    assert "d1" in top_ids or "d2" in top_ids

    # 验证返回格式：(chunk_id, doc_text, rrf_score, source_hint)
    for cid, doc, score, source in fused:
        assert isinstance(cid, str)
        assert isinstance(doc, str)
        assert isinstance(score, float)
        assert source in ("bm25+vector", "vector", "bm25")


def test_rrf_fusion_only_bm25():
    bm25_results = [("d1", "doc1", 5.0)]
    vector_results = []
    fused = rrf_fusion(bm25_results, vector_results)
    assert len(fused) == 1
    assert fused[0][0] == "d1"
    assert fused[0][3] == "bm25"


def test_rrf_fusion_only_vector():
    bm25_results = []
    vector_results = [("d1", "doc1", 0.9)]
    fused = rrf_fusion(bm25_results, vector_results)
    assert len(fused) == 1
    assert fused[0][0] == "d1"
    assert fused[0][3] == "vector"


def test_rrf_fusion_empty():
    assert rrf_fusion([], []) == []


def test_rrf_fusion_dedup():
    """同一文档被两种方法命中时，RRF 中只出现一次（在主函数中去重）。"""
    bm25_results = [("d1", "相同文档内容", 5.0)]
    vector_results = [("d1", "相同文档内容", 0.9)]
    fused = rrf_fusion(bm25_results, vector_results)
    assert len(fused) == 1
    assert fused[0][3] == "bm25+vector"


def test_rrf_fusion_score_ordering():
    """同时被两种方法命中的结果，应该排在只被一种命中的前面。"""
    bm25_results = [("d1", "只在bm25", 5.0), ("d2", "两者都有", 3.0)]
    vector_results = [("d2", "两者都有", 0.9), ("d3", "只在vector", 0.8)]
    fused = rrf_fusion(bm25_results, vector_results)
    # d2 同时被两种方法命中，分数应该最高
    assert fused[0][0] == "d2"
    assert fused[0][3] == "bm25+vector"


# ---------------------------------------------------------------------------
# 5. 动态 Top-K
# ---------------------------------------------------------------------------
def test_dynamic_top_k_simple_query():
    """简单查询应该返回较少的 k"""
    k = compute_dynamic_top_k("你好")
    assert k >= 3
    assert k <= 15


def test_dynamic_top_k_complex_query():
    """复杂查询应该返回较多的 k"""
    k = compute_dynamic_top_k(
        "对比分析Flink和Spark在exactly-once语义实现上的区别，包括checkpoint机制和shuffle设计"
    )
    assert k >= 3


def test_dynamic_top_k_empty():
    k = compute_dynamic_top_k("")
    assert k >= 3  # 不应低于最小值


def test_dynamic_top_k_respects_bounds():
    """即使查询很长，也不应超出 [min_k, max_k] 范围"""
    long_query = "对比分析 " * 50 + "Spark Flink Hive SQL 原理 设计 方案 权衡 为什么 如何"
    k = compute_dynamic_top_k(long_query)
    assert k >= 3
    assert k <= 15


# ---------------------------------------------------------------------------
# 6. 多轮对话检索增强
# ---------------------------------------------------------------------------
def test_multi_turn_basic():
    history = [
        {"role": "user", "content": "Spark的shuffle机制是什么？"},
        {"role": "assistant", "content": "Shuffle是..."},
        {"role": "user", "content": "那窄依赖呢？"},
    ]
    result = build_multi_turn_search_context("再详细说说", history)
    assert "Spark" in result or "shuffle" in result
    assert "再详细说说" in result


def test_multi_turn_no_history():
    result = build_multi_turn_search_context("什么是数仓", None)
    assert result == "什么是数仓"


def test_multi_turn_empty_query():
    assert build_multi_turn_search_context("", []) == ""


def test_multi_turn_max_length():
    history = [{"role": "user", "content": "x" * 500}] * 10
    result = build_multi_turn_search_context("query", history, max_total_length=200)
    assert len(result) <= 200


def test_multi_turn_only_user_messages():
    history = [
        {"role": "assistant", "content": "助手的回复不应出现在检索query中"},
        {"role": "user", "content": "Flink checkpoint 原理"},
        {"role": "assistant", "content": "checkpoint是..."},
        {"role": "user", "content": "Kafka事务怎么用"},
    ]
    result = build_multi_turn_search_context("当前问题", history)
    assert "助手的回复不应出现在检索query中" not in result


# ---------------------------------------------------------------------------
# 7. 向量检索（mock Chroma，不需要真实 API）
# ---------------------------------------------------------------------------
def test_vector_search_empty_store():
    """知识库为空时应返回空列表"""
    with patch("rag_engine.is_vector_store_empty", return_value=True):
        results = vector_search_with_scores("测试查询")
        assert results == []


def test_vector_search_no_api():
    """API 未配置时应返回空列表"""
    with patch("rag_engine.is_vector_store_empty", return_value=False), \
         patch("rag_engine.config.is_api_configured", return_value=False):
        results = vector_search_with_scores("测试查询")
        assert results == []


# ---------------------------------------------------------------------------
# 8. 索引健康检查（mock Chroma）
# ---------------------------------------------------------------------------
def test_check_index_health_empty():
    with patch("rag_engine._persistent_chroma") as mock_chroma, \
         patch("rag_engine.project_data_dir") as mock_dir:
        # 模拟空的 Chroma 集合
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_dir.return_value = MagicMock()
        mock_dir.return_value.is_dir.return_value = True
        mock_dir.return_value.resolve.return_value = mock_dir.return_value
        mock_dir.return_value.glob.return_value = []

        report = check_index_health()
        assert report["ok"] is True
        assert report["checks"]["chroma_chunks"] == 0


def test_get_index_stats():
    with patch("rag_engine.get_collection") as mock_col:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_collection.get.return_value = {
            "ids": ["d1", "d2"],
            "metadatas": [
                {"kb_category": "学习资料"},
                {"kb_category": "面试面经"},
            ],
        }
        mock_col.return_value = mock_collection

        stats = get_index_stats()
        assert stats["total_chunks"] == 42
        assert "学习资料" in stats["by_category"]
        assert "面试面经" in stats["by_category"]


# ---------------------------------------------------------------------------
# 9. 加载文档（mock Chroma）
# ---------------------------------------------------------------------------
def test_load_all_documents_empty():
    with patch("rag_engine.get_collection") as mock_col:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_col.return_value = mock_collection

        ids, docs = _load_all_documents_for_bm25()
        assert ids == []
        assert docs == []
