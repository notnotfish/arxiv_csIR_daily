# ArXiv cs.IR 每日论文摘要

自动化抓取 arXiv cs.IR (Information Retrieval) 分类的最新论文，使用 DeepSeek 进行翻译和摘要生成。

## 功能特性

- 每日自动从 arXiv 获取最近 30 天的 cs.IR 论文
- 基于关键词智能筛选（支持词形还原，如 ranking → rank）
- 使用 DeepSeek API 进行：
  - 标题中文翻译
  - 摘要总结（3-4句中文）
  - 所属公司/机构识别
  - 兴趣度评分（1-5分）及理由
- 生成结构化的 Markdown 报告
- 通过 GitHub Actions 每日自动运行

### 技术亮点

- **安全的配置管理**：支持环境变量和本地配置覆盖，避免 API Key 泄露
- **智能关键词匹配**：改进的词形还原逻辑，支持多词性匹配
- **健壮的 API 调用**：自动重试机制和智能 JSON 解析
- **并发处理**：多线程并发分析论文，提升处理速度
- **完善的日志**：结构化日志输出，便于调试和监控

## 快速开始

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（三选一）
# 方式1: 命令行传入（最方便，推荐用于测试）
python arxiv_fetcher.py --api-key sk-your-api-key-here

# 方式2: 设置环境变量
export DEEPSEEK_API_KEY='your-api-key-here'
python arxiv_fetcher.py

# 方式3: 创建本地配置文件
cp config.local.yaml.example config.local.yaml
# 编辑 config.local.yaml 并填入你的 API Key
python arxiv_fetcher.py

# 查看所有命令行选项
python arxiv_fetcher.py --help
```

### GitHub Actions 配置

1. Fork 本仓库
2. 在 GitHub 仓库设置中添加 Secret：
   - 名称：`DEEPSEEK_API_KEY`
   - 值：你的 DeepSeek API Key
3. GitHub Actions 将每天 UTC 02:00（北京时间 10:00）自动运行
4. 也可在 Actions 页面手动触发运行

## 配置说明

### 配置文件层级

API Key 配置优先级（从高到低）：

1. **命令行参数** `--api-key`（最高优先级）
   - 适用于临时测试和本地开发
   - `python arxiv_fetcher.py --api-key sk-xxx`

2. **环境变量** `DEEPSEEK_API_KEY`
   - 适用于 CI/CD 和生产环境
   - `export DEEPSEEK_API_KEY=sk-xxx`

3. **config.local.yaml** - 本地覆盖配置（不提交到 Git）
   - 用于本地开发时覆盖默认配置
   - 可直接设置 API Key 或其他个性化配置
   - 参考 `config.local.yaml.example` 创建

4. **config.yaml** - 主配置文件（提交到 Git）
   - 包含默认配置
   - API Key 使用环境变量占位符 `${DEEPSEEK_API_KEY}`

### 可配置项

编辑 `config.yaml` 或 `config.local.yaml`：

- **关键词**：公司名称和技术术语（支持词形还原）
- **DeepSeek API**：
  - `api_key`: API 密钥
  - `temperature`: 生成温度（0-1）
  - `max_tokens`: 最大生成 token 数
  - `max_retries`: API 失败重试次数
  - `retry_delay`: 重试间隔（秒）
  - `max_workers`: 并发分析线程数
- **报告设置**：Top N 篇论文、最大作者数、输出目录等

## 输出示例

报告保存在 `summary/YYYYMM/YYYYMMDD_HHMMSS.md`，包含：

- 英文/中文标题
- 作者（最多4个）
- 提交时间
- 文章链接
- 所属公司
- 命中关键词及数量
- 兴趣评分及理由
- 摘要总结（3-4句中文）
