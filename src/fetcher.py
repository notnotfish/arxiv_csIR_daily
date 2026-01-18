"""ArXiv论文抓取器"""
import re
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import List, Dict

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class ArxivFetcher:
    """ArXiv论文抓取器"""

    def __init__(self, config: Dict, target_timezone: timezone):
        """
        初始化抓取器

        Args:
            config: 配置字典
            target_timezone: 目标时区
        """
        self.config = config
        self.timezone = target_timezone
        self.lemmatizer = None

        # 初始化词形还原器
        if self.config['matching'].get('use_lemmatization', True) and NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            # 确保下载了必要的NLTK数据
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                try:
                    nltk.download('wordnet', quiet=True)
                    nltk.download('omw-1.4', quiet=True)
                except Exception as e:
                    logger.warning(f"下载NLTK数据失败: {e}")

    def fetch_papers(self) -> List[Dict]:
        """从ArXiv获取论文"""
        # 计算最近30天的日期范围（使用配置的时区）
        end_date = datetime.now(self.timezone)
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

    def rank_papers(self, papers: List[Dict]) -> List[Dict]:
        """按关键词数量排序论文"""
        # 过滤掉没有匹配关键词的论文
        filtered = [p for p in papers if p['keyword_count'] > 0]

        # 按关键词数量降序排序
        sorted_papers = sorted(filtered, key=lambda x: x['keyword_count'], reverse=True)

        # 取top N
        top_n = self.config['report']['top_n']
        return sorted_papers[:top_n]

    def _get_keywords(self) -> List[str]:
        """获取所有关键词"""
        keywords = []
        keywords.extend(self.config['keywords'].get('company', []))
        keywords.extend(self.config['keywords'].get('technical', []))
        return keywords

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
