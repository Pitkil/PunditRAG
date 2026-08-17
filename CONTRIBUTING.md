# Contributing to PunditRAG

感谢你愿意改进 PunditRAG。提交改动前，请先确认问题可以复现，并尽量让修改保持小而清晰。

## 开发环境

1. 安装 Python 3.11、uv、Docker Desktop 和 Docker Compose。
2. 复制 `.env.docker.example` 为 `.env.docker`，填写本地密钥与服务密码。
3. 首次使用 `.\start.ps1 -Build` 构建并启动；后续使用 `.\start.ps1` 直接启动现有镜像。
4. 不要提交 `.env.docker`、模型、缓存、日志、上传文档或第三方评测原始数据。

## 提交改动

- Bug 修复应包含能够覆盖问题的回归测试。
- 检索、切分、重排和 Prompt 修改应说明对评测口径与结果的影响。
- Prompt 修改必须保持调用方占位符和输出解析契约，并补充或更新 Prompt 渲染测试与行为回归测试。
- 不要用硬编码答案、人工覆盖结果或跨样本共享知识库来提高评测分数。
- 新增环境变量时，同步更新 `.env.example`、`.env.docker.example` 和 README。
- 新增依赖前确认其许可证允许在 MIT 项目中使用。

## 本地检查

```powershell
$tests = @(
  "16_node_rerank.py",
  "17_text_compress_utils.py",
  "18_node_answer_output.py",
  "19_workspace_features.py",
  "20_rag_reliability.py",
  "21_reliability_hardening.py"
)

foreach ($test in $tests) {
  .\.venv\Scripts\python.exe (Join-Path "test" $test)
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

.\.venv\Scripts\python.exe -m compileall -q app eval test
git diff --check
```

## Pull Request

PR 描述应包含：

- 问题背景和修改范围
- 关键实现选择
- 测试与评测结果
- 配置、API 或兼容性变化
- 尚未覆盖的风险

请勿在公开 Issue 或 PR 中粘贴 API Key、访问令牌、私有文档或生产日志。
