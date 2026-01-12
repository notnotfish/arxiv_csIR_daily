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

## 快速开始

### 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 运行脚本
python arxiv_fetcher.py
```

### GitHub Actions 配置

1. Fork 本仓库
2. 在 GitHub 仓库设置中添加 Secret：
   - 名称：`DEEPSEEK_API_KEY`
   - 值：你的 DeepSeek API Key
3. GitHub Actions 将每天 UTC 02:00（北京时间 10:00）自动运行
4. 也可在 Actions 页面手动触发运行

## 配置说明

编辑 `config.yaml` 可配置：

- **关键词**：公司名称和技术术语
- **DeepSeek API**：API Key、模型参数
- **报告设置**：Top N 篇论文、输出目录等

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
