# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated ArXiv cs.IR (Information Retrieval) daily paper aggregator designed for recommendation system engineers. The system fetches papers from arXiv, filters by relevance using configurable keywords, and uses DeepSeek API for Chinese translation and summarization.

## Architecture

### Project Structure (Modularized)

```
arxiv_csIR_daily/
├── arxiv_fetcher.py          # Main entry point (130 lines, simplified)
├── src/                      # Source modules
│   ├── __init__.py           # Package exports
│   ├── models.py             # Data models (Paper class)
│   ├── utils.py              # Utilities (logging, config loading)
│   ├── fetcher.py            # ArxivFetcher class
│   ├── analyzer.py           # DeepSeekAnalyzer class
│   └── report.py             # ReportGenerator and WebGenerator
├── config.yaml               # Configuration file
└── summary/                  # Generated reports
```

### Core Components

1. **arxiv_fetcher.py** - Lightweight main entry point (~130 lines)
   - Parses command-line arguments
   - Orchestrates the workflow by calling src modules
   - Handles initialization and error handling

2. **src/fetcher.py** - `ArxivFetcher` class
   - Fetches papers from arXiv API
   - Performs keyword matching with NLTK lemmatization support
   - Ranks and filters papers by keyword relevance

3. **src/analyzer.py** - `DeepSeekAnalyzer` class
   - Calls DeepSeek API for Chinese translation and summarization
   - Provides interest scoring (1-5) for recommendation engineers
   - Handles concurrent analysis with retry logic

4. **src/report.py** - Report generation
   - `ReportGenerator`: Creates Markdown and HTML reports
   - `WebGenerator`: Generates GitHub Pages index.html

5. **src/utils.py** - Utilities
   - Configuration loading with environment variable substitution
   - Timezone-aware logging setup
   - Configuration validation

6. **src/models.py** - Data models
   - `Paper` dataclass for type-safe paper data

7. **config.yaml** - Centralized configuration:
   - arXiv API settings (category, date range, max results)
   - DeepSeek API credentials and model parameters
   - Keywords grouped by type (company, technical)
   - Report output settings
   - Timezone configuration (UTC offset)

8. **GitHub Actions workflow** (`.github/workflows/daily_arxiv_fetch.yml`)
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

- **Modular Architecture**: Code split into focused modules for better maintainability
- **Timezone Management**: Configurable timezone (default UTC+8 Shanghai) for all timestamps
- **Date Range**: Automatically calculates last 30 days from configured timezone
- **Author Display**: Limited to 4 authors in report, append "等N人" for more
- **Sorting**: Final report sorted by (keyword_count, interest_score) descending
- **Error Handling**: DeepSeek API failures return minimal fallback data without crashing
- **File Organization**: Reports stored in `summary/YYYYMM/YYYYMMDD_HHMMSS.md` format
- **Backward Compatibility**: `arxiv_fetcher.py` remains as main entry point
