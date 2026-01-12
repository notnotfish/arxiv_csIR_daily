# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated ArXiv cs.IR (Information Retrieval) daily paper aggregator designed for recommendation system engineers. The system fetches papers from arXiv, filters by relevance using configurable keywords, and uses DeepSeek API for Chinese translation and summarization.

## Architecture

### Core Components

1. **arxiv_fetcher.py** - Main entry point containing three classes:
   - `ArxivFetcher`: Fetches papers from arXiv API, performs keyword matching with lemmatization support
   - `DeepSeekAnalyzer`: Calls DeepSeek API for translation, summarization, and relevance scoring
   - `ReportGenerator`: Generates formatted markdown reports and saves to dated directories

2. **config.yaml** - Centralized configuration for:
   - arXiv API settings (category, date range, max results)
   - DeepSeek API credentials and model parameters
   - Keywords grouped by type (company, technical)
   - Report output settings

3. **GitHub Actions workflow** (`.github/workflows/daily_arxiv_fetch.yml`)
   - Scheduled to run daily at UTC 02:00 (Beijing 10:00)
   - Manually triggerable via `workflow_dispatch`
   - Commits generated reports back to repository

### Data Flow

```
arXiv API → Keyword Matching (top 20) → DeepSeek Analysis → Markdown Report → summary/YYYYMM/YYYYMMDD_HHMMSS.md
```

### Keyword Matching Logic

- Uses NLTK lemmatization to handle word variations (e.g., "ranking" → "rank")
- Case-insensitive matching with word boundaries
- Keywords categorized as company names or technical terms
- Papers ranked by keyword match count, then top N selected

## Common Commands

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the fetcher locally
python arxiv_fetcher.py

# Download NLTK data manually (if needed)
python -c "import nltk; nltk.download('wordnet')"
```

### Configuration

- Edit `config.yaml` to modify keywords, API settings, or report parameters
- Add `config.local.yaml` to override settings locally (gitignored)
- For GitHub Actions, set `DEEPSEEK_API_KEY` in repository Secrets

## Key Implementation Details

- **Date Range**: Automatically calculates last 30 days from current UTC time
- **Author Display**: Limited to 4 authors in report, append "等N人" for more
- **Sorting**: Final report sorted by (keyword_count, interest_score) descending
- **Error Handling**: DeepSeek API failures return minimal fallback data without crashing
- **File Organization**: Reports stored in `summary/YYYYMM/YYYYMMDD_HHMMSS.md` format
