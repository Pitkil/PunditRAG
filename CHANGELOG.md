# Changelog

本项目遵循 [Semantic Versioning](https://semver.org/)。

## [Unreleased]

### Added

- 文档导入、知识库管理和查询工作台
- Dense/Sparse、HyDE、RRF 与 BGE Reranker 检索链路
- 带来源引用的证据约束回答
- 自建集、RGB、CRUD-RAG、MTRAG 评测适配器
- 上传、CORS、私有对象存储和健康检查加固
- MIT License、贡献指南和安全策略

### Changed

- 技术指标使用带标题的密集行分组切片
- 重排至少保留两条合格证据，最低相关度调整为 0.09
- 重排没有合格证据但仍有召回时，将前三条低置信候选交给回答模型核验；只有完全零召回才提前拒答
- 统一规范 9 个 Prompt 的注入防护、精确信息保留、部分回答、冲突处理和输出契约
- 评测使用独立会话并计算真实 P50、P95、P99

## [0.1.0] - 2026-08-17

- 首个可公开发布的开发版本。
