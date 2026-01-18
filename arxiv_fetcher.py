"""
ArXiv论文抓取和分析工具
每日自动抓取cs.IR领域论文，使用DeepSeek进行翻译和摘要
"""

import os
import re
import yaml
import json
import time
import logging
import argparse
import requests
import markdown
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """论文数据结构"""
    title: str
    translated_title: str = ""
    authors: List[str] = field(default_factory=list)
    submitted_date: str = ""
    url: str = ""
    abstract: str = ""
    company: str = ""
    summary: str = ""
    matched_keywords: List[str] = field(default_factory=list)
    keyword_count: int = 0
    interest_score: int = 0
    interest_reason: str = ""


class ArxivFetcher:
    """ArXiv论文抓取器"""

    def __init__(self, config_path: str = "config.yaml"):
        """初始化配置"""
        self.config = self._load_config(config_path)
        self._validate_config()
        self.lemmatizer = None
        if self.config['matching'].get('use_lemmatization', True) and NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            # 确保下载了必要的NLTK数据
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                try:
                    nltk.download('wordnet', quiet=True)
                    nltk.download('omw-1.4', quiet=True)  # 多语言WordNet
                except Exception as e:
                    logger.warning(f"下载NLTK数据失败: {e}")

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件，支持本地覆盖和环境变量替换"""
        # 加载主配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 尝试加载本地覆盖配置
        local_config_path = config_path.replace('.yaml', '.local.yaml')
        if os.path.exists(local_config_path):
            logger.info(f"检测到本地配置文件: {local_config_path}")
            with open(local_config_path, 'r', encoding='utf-8') as f:
                local_config = yaml.safe_load(f)
                if local_config:
                    self._deep_merge(config, local_config)

        # 替换环境变量
        config = self._replace_env_vars(config)

        return config

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """深度合并字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _replace_env_vars(self, obj: Any) -> Any:
        """递归替换配置中的环境变量"""
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # 匹配 ${VAR_NAME} 或 $VAR_NAME 格式
            pattern = r'\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)'
            def replace_match(match):
                var_name = match.group(1) or match.group(2)
                return os.environ.get(var_name, match.group(0))
            return re.sub(pattern, replace_match, obj)
        else:
            return obj

    def _validate_config(self) -> None:
        """验证配置完整性"""
        required_fields = {
            'arxiv': ['base_url', 'category', 'max_results'],
            'deepseek': ['api_key', 'base_url', 'model'],
            'keywords': ['company', 'technical'],
            'report': ['top_n', 'max_authors', 'output_dir']
        }

        for section, fields in required_fields.items():
            if section not in self.config:
                raise ValueError(f"配置缺少必要的section: {section}")
            for field in fields:
                if field not in self.config[section]:
                    raise ValueError(f"配置缺少必要的字段: {section}.{field}")

        # 验证 API Key
        api_key = self.config['deepseek']['api_key']
        if not api_key or api_key.startswith('$'):
            raise ValueError(
                "DeepSeek API Key 未设置！\n"
                "请通过以下方式之一设置：\n"
                "1. 设置环境变量: export DEEPSEEK_API_KEY='your-key'\n"
                "2. 创建 config.local.yaml 并设置 deepseek.api_key"
            )

        logger.info("配置验证通过")

    def _get_keywords(self) -> List[str]:
        """获取所有关键词"""
        keywords = []
        keywords.extend(self.config['keywords'].get('company', []))
        keywords.extend(self.config['keywords'].get('technical', []))
        return keywords

    def _get_wordnet_pos(self, word: str) -> str:
        """根据词尾特征判断词性"""
        # 简单的启发式规则
        if word.endswith('ing') or word.endswith('ed') or word.endswith('ize'):
            return wordnet.VERB
        elif word.endswith('ly'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def _normalize_word(self, word: str) -> str:
        """标准化单词（小写、词形还原）"""
        word = word.lower()
        if self.lemmatizer and NLTK_AVAILABLE:
            try:
                # 尝试作为动词和名词还原，取最短的形式
                noun_form = self.lemmatizer.lemmatize(word, pos=wordnet.NOUN)
                verb_form = self.lemmatizer.lemmatize(word, pos=wordnet.VERB)
                word = min([noun_form, verb_form], key=len)
            except Exception as e:
                logger.debug(f"词形还原失败 {word}: {e}")
        return word

    def _extract_keywords_from_text(self, text: str, keywords: List[str]) -> List[str]:
        """从文本中提取匹配的关键词"""
        if not text:
            return []

        content = text.lower()
        matched = []

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # 如果启用词形还原，则对文本中的每个词进行还原后匹配
            if self.lemmatizer and NLTK_AVAILABLE:
                normalized_keyword = self._normalize_word(keyword_lower)
                # 使用单词边界匹配原始关键词
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                if re.search(pattern, content):
                    matched.append(keyword)
                    continue

                # 在文本中查找可能匹配的词（通过词形还原）
                # 提取文本中的所有单词
                words_in_text = re.findall(r'\b\w+\b', content)
                for word in words_in_text:
                    if self._normalize_word(word) == normalized_keyword:
                        matched.append(keyword)
                        break
            else:
                # 不使用词形还原，直接精确匹配
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                if re.search(pattern, content):
                    matched.append(keyword)

        return matched

    def fetch_papers(self) -> List[Dict]:
        """从ArXiv获取论文"""
        # 计算最近30天的日期范围
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=30)

        # ArXiv API格式：YYYYMMDDHHMM
        start_str = start_date.strftime("%Y%m%d%H%M")
        end_str = end_date.strftime("%Y%m%d%H%M")

        # 构建查询URL
        category = self.config['arxiv']['category']
        max_results = self.config['arxiv']['max_results']

        query = f"cat:{category}+AND+submittedDate:[{start_str}+TO+{end_str}]"
        url = (
            f"{self.config['arxiv']['base_url']}?"
            f"search_query={query}&"
            f"sortBy=submittedDate&"
            f"sortOrder=descending&"
            f"start=0&"
            f"max_results={max_results}"
        )

        logger.info(f"正在从ArXiv获取论文...")
        logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

        # arXiv API 要求设置 User-Agent
        headers = {
            'User-Agent': 'ArxivBot/1.0 (mailto:your-email@example.com)'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        return self._parse_arxiv_response(response.text)

    def _parse_arxiv_response(self, xml_text: str) -> List[Dict]:
        """解析ArXiv API返回的XML"""
        root = ET.fromstring(xml_text)

        # ArXiv使用命名空间
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }

        papers = []
        keywords = self._get_keywords()

        for entry in root.findall('atom:entry', ns):
            # 提取标题
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""

            # 提取作者
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text)

            # 提取提交时间
            published_elem = entry.find('atom:published', ns)
            submitted_date = published_elem.text if published_elem is not None else ""

            # 提取链接
            id_elem = entry.find('atom:id', ns)
            paper_url = id_elem.text if id_elem is not None else ""

            # 提取摘要
            summary_elem = entry.find('atom:summary', ns)
            abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""

            # 匹配关键词（在标题和摘要中搜索）
            text_to_search = f"{title} {abstract}"
            matched_keywords = self._extract_keywords_from_text(text_to_search, keywords)

            papers.append({
                'title': title,
                'authors': authors,
                'submitted_date': submitted_date,
                'url': paper_url,
                'abstract': abstract,
                'matched_keywords': matched_keywords,
                'keyword_count': len(matched_keywords)
            })

        logger.info(f"获取到 {len(papers)} 篇论文")
        return papers

    def rank_papers(self, papers: List[Dict]) -> List[Dict]:
        """按关键词数量排序论文"""
        # 过滤掉没有匹配关键词的论文
        filtered = [p for p in papers if p['keyword_count'] > 0]

        # 按关键词数量降序排序
        sorted_papers = sorted(filtered, key=lambda x: x['keyword_count'], reverse=True)

        # 取top N
        top_n = self.config['report']['top_n']
        return sorted_papers[:top_n]


class DeepSeekAnalyzer:
    """使用DeepSeek API分析论文"""

    def __init__(self, config: Dict):
        self.config = config['deepseek']
        self.api_key = self.config['api_key']
        self.base_url = self.config['base_url']
        self.model = self.config.get('model', 'deepseek-chat')
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 2)

    def _extract_json_from_text(self, text: str) -> Dict:
        """从文本中提取JSON，处理各种格式"""
        text = text.strip()

        # 移除markdown代码块
        if '```' in text:
            # 提取代码块内容
            parts = text.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:  # 代码块内容在奇数位置
                    # 移除可能的语言标识符（如 json）
                    lines = part.strip().split('\n')
                    if lines[0].strip().lower() in ['json', 'jsonc']:
                        part = '\n'.join(lines[1:])
                    text = part.strip()
                    break

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取JSON对象（使用栈匹配括号，支持任意嵌套）
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        pass

        # 如果都失败了，抛出异常
        raise ValueError(f"无法从文本中提取有效的JSON: {text[:100]}...")

    def analyze_paper(self, paper: Dict, keywords: List[str]) -> Dict:
        """分析单篇论文（带重试）"""
        for attempt in range(self.max_retries):
            try:
                return self._analyze_paper_once(paper, keywords)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"分析论文失败 (尝试 {attempt + 1}/{self.max_retries}): "
                        f"{paper['title'][:50]}... - {str(e)}"
                    )
                    time.sleep(self.retry_delay * (attempt + 1))  # 指数退避
                else:
                    logger.error(f"分析论文最终失败: {paper['title'][:50]}... - {str(e)}")
                    return self._get_fallback_analysis(paper, str(e))

    def _analyze_paper_once(self, paper: Dict, keywords: List[str]) -> Dict:
        """分析单篇论文（单次尝试）"""
        title = paper['title']
        abstract = paper['abstract']
        keyword_str = ', '.join(paper['matched_keywords'])

        prompt = f"""你是一位资深的推荐系统算法工程师，专门研究重排、价值建模（LTV）、因果推断等方向。

请分析以下论文，并以JSON格式返回结果：

论文标题：{title}

论文摘要：{abstract}

命中的关键词：{keyword_str}

请返回以下JSON格式的结果（不要有任何其他文字，只要纯JSON）：
{{
  "translated_title": "中文标题翻译",
  "company": "根据作者 affiliation 或内容判断所属公司/机构（如果无法判断则写'未知'）",
  "summary": "用3-4句中文总结论文的核心内容、方法和创新点（不要太短！）",
  "interest_score": 1-5的整数评分，表示这位推荐系统算法工程师对该论文的兴趣程度，
  "interest_reason": "一句话中文说明评分理由"
}}

评分标准：
- 5分：直接涉及重排、LTV、因果推断、ctr/cvr预估等核心推荐技术，且来自知名公司
- 4分：涉及推荐系统的核心技术（召回、排序、多样性等），有较好的创新性
- 3分：与推荐系统相关，但不是核心方向
- 2分：与信息检索相关，但与推荐系统关系较弱
- 1分：与推荐系统基本无关
"""

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ArxivBot/1.0"
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.get('temperature', 0.3),
                "max_tokens": self.config.get('max_tokens', 2000)
            },
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        content = result['choices'][0]['message']['content'].strip()

        # 使用改进的JSON提取方法
        analysis = self._extract_json_from_text(content)

        # 验证必要字段
        required_fields = ['translated_title', 'company', 'summary', 'interest_score', 'interest_reason']
        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"API返回的JSON缺少字段: {field}")

        return analysis

    def _get_fallback_analysis(self, paper: Dict, error_msg: str) -> Dict:
        """返回失败时的默认分析结果"""
        return {
            'translated_title': f"[翻译失败] {paper['title']}",
            'company': '未知',
            'summary': f"摘要生成失败: {error_msg}",
            'interest_score': 1,
            'interest_reason': 'API调用失败'
        }

    def analyze_papers_concurrent(self, papers: List[Dict], keywords: List[str], max_workers: int = 5) -> List[Dict]:
        """并发分析多篇论文"""
        logger.info(f"开始使用DeepSeek分析论文（并发数: {max_workers}）...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(self.analyze_paper, paper, keywords): paper
                for paper in papers
            }

            # 收集结果（保持原始顺序）
            analyzed_papers = []
            for idx, future in enumerate(as_completed(future_to_paper), 1):
                paper = future_to_paper[future]
                try:
                    analysis = future.result()
                    # 创建新字典而不是修改原字典，避免线程安全问题
                    analyzed_paper = {**paper, **analysis}
                    analyzed_papers.append(analyzed_paper)
                    logger.info(f"[{idx}/{len(papers)}] 完成分析: {paper['title'][:60]}...")
                except Exception as e:
                    logger.error(f"并发分析出错: {paper['title'][:50]}... - {e}")
                    # 添加失败的默认结果
                    analyzed_paper = {**paper, **self._get_fallback_analysis(paper, str(e))}
                    analyzed_papers.append(analyzed_paper)

        return analyzed_papers


class ReportGenerator:
    """报告生成器"""

    def __init__(self, config: Dict):
        self.config = config

    def generate(self, analyzed_papers: List[Dict], run_time: datetime) -> str:
        """生成Markdown报告"""
        top_n = self.config['report']['top_n']
        max_authors = self.config['report']['max_authors']

        # 按关键词数量 + 兴趣分数排序
        sorted_papers = sorted(
            analyzed_papers,
            key=lambda x: (x['keyword_count'], x.get('interest_score', 0)),
            reverse=True
        )

        date_str = run_time.strftime("%Y年%m月%d日 %H:%M:%S")

        markdown = f"""# ArXiv cs.IR 每日论文摘要

**生成时间**: {date_str}
**论文数量**: {len(sorted_papers)} 篇（Top {top_n}）
**排序规则**: 按关键词命中数 + 兴趣评分排序

---

"""

        for idx, paper in enumerate(sorted_papers, 1):
            # 处理作者列表
            all_authors = paper.get('authors', [])
            authors = all_authors[:max_authors]
            if len(all_authors) > max_authors:
                remaining = len(all_authors) - max_authors
                authors.append(f"等{remaining}人")

            author_str = "、".join(authors)

            # 处理时间
            submitted_date = paper.get('submitted_date', '')
            if submitted_date:
                try:
                    dt = datetime.fromisoformat(submitted_date.replace('Z', '+00:00'))
                    date_str = dt.strftime("%Y-%m-%d")
                except (ValueError, IndexError) as e:
                    logger.debug(f"日期解析失败: {submitted_date} - {e}")
                    date_str = submitted_date[:10] if len(submitted_date) >= 10 else submitted_date
            else:
                date_str = "未知"

            # 关键词展示
            keywords = paper.get('matched_keywords', [])
            keyword_str = f"【{len(keywords)}】 {', '.join(keywords)}"

            # 兴趣评分
            score = paper.get('interest_score', 0)
            reason = paper.get('interest_reason', '')
            interest_str = f"【{score}分】 {reason}"

            markdown += f"""## {idx}. {paper.get('translated_title', paper['title'])}

**英文标题**: {paper['title']}

**作者**: {author_str}

**提交时间**: {date_str}

**文章链接**: [查看原文]({paper['url']})

**所属公司**: {paper.get('company', '未知')}

**命中关键词**: {keyword_str}

**感兴趣评分**: {interest_str}

**摘要总结**:
{paper.get('summary', paper.get('abstract', ''))}

---

"""

        return markdown

    def save_report(self, content: str, run_time: datetime) -> str:
        """保存报告到文件"""
        output_dir = Path(self.config['report']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建YYYYMM子目录
        yyyymm = run_time.strftime("%Y%m")
        month_dir = output_dir / yyyymm
        month_dir.mkdir(parents=True, exist_ok=True)

        # 文件名：YYYYMMDD_HHMMSS.md
        filename = run_time.strftime("%Y%m%d_%H%M%S.md")
        filepath = month_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"报告已保存到: {filepath}")
        return str(filepath)

    def generate_html(self, content: str, run_time: datetime) -> str:
        """将Markdown内容转换为带样式的HTML"""
        date_str = run_time.strftime("%Y年%m月%d日 %H:%M:%S")

        # 转换Markdown到HTML
        md = markdown.Markdown(extensions=[
            'tables',           # 表格支持
            'fenced_code',      # 代码块支持
            'nl2br',           # 换行支持
            'sane_lists'       # 更好的列表处理
        ])
        html_content = md.convert(content)

        # HTML模板
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArXiv cs.IR 每日论文摘要 - {date_str}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            background-color: #f6f8fa;
            padding: 20px;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        }}

        h1 {{
            font-size: 2em;
            margin-bottom: 0.5em;
            padding-bottom: 0.3em;
            border-bottom: 2px solid #eaecef;
            color: #0366d6;
        }}

        h2 {{
            font-size: 1.5em;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            padding-bottom: 0.2em;
            border-bottom: 1px solid #eaecef;
        }}

        h3 {{
            font-size: 1.25em;
            margin-top: 1em;
            margin-bottom: 0.5em;
        }}

        p {{
            margin-bottom: 1em;
        }}

        strong {{
            font-weight: 600;
        }}

        a {{
            color: #0366d6;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        hr {{
            border: 0;
            border-top: 1px solid #eaecef;
            margin: 2em 0;
        }}

        code {{
            background-color: #f6f8fa;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 0.9em;
        }}

        pre {{
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
            margin-bottom: 1em;
        }}

        pre code {{
            background-color: transparent;
            padding: 0;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 1em;
        }}

        table th,
        table td {{
            border: 1px solid #dfe2e5;
            padding: 6px 13px;
        }}

        table th {{
            background-color: #f6f8fa;
            font-weight: 600;
        }}

        .footer {{
            margin-top: 3em;
            padding-top: 1em;
            border-top: 1px solid #eaecef;
            text-align: center;
            color: #586069;
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}

            body {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
        <div class="footer">
            <p><a href="../../index.html">← 返回主页</a></p>
            <p>生成时间: {date_str}</p>
        </div>
    </div>
</body>
</html>"""

        return html

    def save_html_report(self, html_content: str, run_time: datetime) -> str:
        """保存HTML报告到文件"""
        output_dir = Path(self.config['report']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建YYYYMM子目录
        yyyymm = run_time.strftime("%Y%m")
        month_dir = output_dir / yyyymm
        month_dir.mkdir(parents=True, exist_ok=True)

        # 文件名：YYYYMMDD_HHMMSS.html
        filename = run_time.strftime("%Y%m%d_%H%M%S.html")
        filepath = month_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML报告已保存到: {filepath}")
        return str(filepath)


class WebGenerator:
    """网页生成器 - 生成GitHub Pages主页"""

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['report']['output_dir'])

    def _scan_reports(self) -> List[Dict[str, str]]:
        """扫描所有报告文件（包括 .md 和 .html）"""
        reports = []

        if not self.output_dir.exists():
            logger.warning(f"输出目录不存在: {self.output_dir}")
            return reports

        # 扫描所有 .html 文件
        for html_file in self.output_dir.glob("*/*.html"):
            try:
                # 解析文件名：YYYYMMDD_HHMMSS.html
                filename = html_file.stem
                date_str, time_str = filename.split('_')

                # 解析时间
                year = date_str[:4]
                month = date_str[4:6]
                day = date_str[6:8]
                hour = time_str[:2]
                minute = time_str[2:4]
                second = time_str[4:6]

                dt = datetime(
                    int(year), int(month), int(day),
                    int(hour), int(minute), int(second)
                )

                reports.append({
                    'path': str(html_file.relative_to(self.output_dir.parent)),
                    'datetime': dt,
                    'title': f"{dt.strftime('%Y年%m月%d日 %H:%M:%S')} 论文摘要"
                })
            except (ValueError, IndexError) as e:
                logger.warning(f"跳过文件 {html_file}: 文件名格式不正确 - {e}")
                continue

        # 按时间降序排序
        reports.sort(key=lambda x: x['datetime'], reverse=True)
        return reports

    def generate_index(self) -> str:
        """生成主页 index.html"""
        reports = self._scan_reports()

        # 获取最新报告
        latest_report = reports[0] if reports else None

        # 最近10条报告
        recent_reports = reports[:10]

        # 生成HTML
        html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArXiv cs.IR 每日论文摘要</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 0.3em;
            color: #0366d6;
            text-align: center;
        }

        .subtitle {
            text-align: center;
            color: #586069;
            margin-bottom: 2em;
            font-size: 1.1em;
        }

        .section {
            margin-top: 2em;
        }

        .section h2 {
            font-size: 1.5em;
            margin-bottom: 1em;
            padding-bottom: 0.5em;
            border-bottom: 2px solid #eaecef;
            color: #0366d6;
        }

        .latest-report {
            background-color: #f6f8fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #0366d6;
        }

        .latest-report h3 {
            margin-bottom: 0.5em;
            color: #0366d6;
        }

        .report-link {
            display: inline-block;
            margin-top: 10px;
            padding: 10px 20px;
            background-color: #0366d6;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            transition: background-color 0.3s;
        }

        .report-link:hover {
            background-color: #0256c7;
        }

        .report-list {
            list-style: none;
        }

        .report-list li {
            margin-bottom: 0.8em;
            padding: 12px;
            background-color: #f6f8fa;
            border-radius: 6px;
            transition: background-color 0.2s;
        }

        .report-list li:hover {
            background-color: #e1e4e8;
        }

        .report-list a {
            color: #0366d6;
            text-decoration: none;
            font-weight: 500;
        }

        .report-list a:hover {
            text-decoration: underline;
        }

        .no-reports {
            color: #586069;
            text-align: center;
            padding: 20px;
            font-style: italic;
        }

        .footer {
            margin-top: 3em;
            padding-top: 1em;
            border-top: 1px solid #eaecef;
            text-align: center;
            color: #586069;
            font-size: 0.9em;
        }

        .footer a {
            color: #0366d6;
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }

            h1 {
                font-size: 2em;
            }

            body {
                padding: 20px 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>ArXiv cs.IR 每日论文摘要</h1>
        <p class="subtitle">自动抓取并分析推荐系统领域最新论文</p>
"""

        if latest_report:
            html += f"""
        <div class="section">
            <h2>最新报告</h2>
            <div class="latest-report">
                <h3>{latest_report['title']}</h3>
                <p>每日自动更新，汇总推荐系统、信息检索领域的最新研究成果</p>
                <a href="{latest_report['path']}" class="report-link">查看最新报告 →</a>
            </div>
        </div>
"""
        else:
            html += """
        <div class="section">
            <p class="no-reports">暂无报告，请等待首次运行</p>
        </div>
"""

        if recent_reports:
            html += """
        <div class="section">
            <h2>历史报告</h2>
            <ul class="report-list">
"""
            for report in recent_reports:
                html += f"""                <li><a href="{report['path']}">{report['title']}</a></li>
"""

            html += """            </ul>
        </div>
"""

        html += """
        <div class="footer">
            <p>由 <a href="https://github.com/anthropics/claude-code" target="_blank">Claude Code</a> 驱动</p>
            <p>数据来源: <a href="https://arxiv.org/" target="_blank">arXiv.org</a></p>
        </div>
    </div>
</body>
</html>"""

        # 保存到项目根目录
        index_path = self.output_dir.parent / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"主页已生成: {index_path}")
        return str(index_path)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='ArXiv cs.IR 论文抓取和分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用环境变量中的 API Key
  python arxiv_fetcher.py

  # 通过命令行传入 API Key
  python arxiv_fetcher.py --api-key sk-xxx

  # 使用自定义配置文件
  python arxiv_fetcher.py --config my_config.yaml --api-key sk-xxx
        """
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='DeepSeek API Key（优先级高于环境变量和配置文件）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )

    args = parser.parse_args()

    # 如果通过命令行传入了 API Key，设置到环境变量中
    if args.api_key:
        os.environ['DEEPSEEK_API_KEY'] = args.api_key
        logger.info("使用命令行传入的 API Key")

    logger.info("=" * 60)
    logger.info("ArXiv cs.IR 论文抓取和分析工具")
    logger.info("=" * 60)

    try:
        # 初始化
        fetcher = ArxivFetcher(args.config)
        analyzer = DeepSeekAnalyzer(fetcher.config)
        reporter = ReportGenerator(fetcher.config)

        run_time = datetime.now(timezone.utc)

        # 1. 获取论文
        papers = fetcher.fetch_papers()
        if not papers:
            logger.warning("未获取到论文，程序退出")
            return

        # 2. 排序并取top N
        top_papers = fetcher.rank_papers(papers)
        logger.info(f"按关键词排序后，选取Top {len(top_papers)} 篇论文")

        # 3. 使用DeepSeek并发分析论文
        keywords = fetcher._get_keywords()
        max_workers = fetcher.config.get('deepseek', {}).get('max_workers', 5)
        analyzed_papers = analyzer.analyze_papers_concurrent(top_papers, keywords, max_workers)

        # 4. 生成报告
        logger.info("生成报告...")
        report_content = reporter.generate(analyzed_papers, run_time)

        # 5. 保存 Markdown 报告
        reporter.save_report(report_content, run_time)

        # 6. 生成并保存 HTML 报告
        logger.info("生成HTML报告...")
        html_content = reporter.generate_html(report_content, run_time)
        reporter.save_html_report(html_content, run_time)

        # 7. 更新主页索引
        logger.info("更新主页索引...")
        web_gen = WebGenerator(fetcher.config)
        web_gen.generate_index()

        logger.info("=" * 60)
        logger.info("完成！")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
