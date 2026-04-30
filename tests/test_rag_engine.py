# -*- coding: utf-8 -*-
"""简单单元测试：验证分块逻辑符合 chunk_size / overlap 约定。"""
from rag_engine import simple_chunk_text


def test_chunk_overlap():
    text = "a" * 100
    parts = simple_chunk_text(text, chunk_size=30, overlap=5)
    assert len(parts) >= 2
    # 相邻块应在 overlap 区域有重复
    assert parts[0][-5:] == parts[1][:5]
