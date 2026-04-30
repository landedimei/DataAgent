# 数据开发（DE）小抄 — 样例（用于 RAG 建索引自测）

## Spark 与批处理
- 宽依赖（shuffle）会触发 stage 切分，要尽量减少 shuffle。
- 面试常问：repartition 与 coalesce 区别；为什么 combineByKey 有时优于 groupByKey。

## 数仓分层
- 常见 ODS / DWD / DWS / ADS：越往上越面向分析主题，越往下越贴源。
- 缓慢变化维（SCD）有哪些类型、拉链表在 Hive 里怎么设计。

## SQL 与性能
- 大表 join 时：分区裁剪、map-side join 条件、数据倾斜的几种缓解手段（随机前缀、两阶段聚合等）。
