"""
ArXiv论文抓取和分析工具
每日自动抓取cs.IR领域论文，使用DeepSeek进行翻译和摘要
"""

import os
import re
import yaml
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import unquote

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


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
        self.lemmatizer = None
        if self.config['matching'].get('use_lemmatization', True) and NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            # 确保下载了必要的NLTK数据
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                try:
                    nltk.download('wordnet', quiet=True)
                except:
                    pass

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _get_keywords(self) -> List[str]:
        """获取所有关键词"""
        keywords = []
        keywords.extend(self.config['keywords'].get('company', []))
        keywords.extend(self.config['keywords'].get('technical', []))
        return keywords

    def _normalize_word(self, word: str) -> str:
        """标准化单词（小写、词形还原）"""
        word = word.lower()
        if self.lemmatizer:
            # 尝试词形还原
            try:
                word = self.lemmatizer.lemmatize(word)
                # 处理ing结尾的词
                if word.endswith('ing'):
                    base = word[:-3]
                    word = self.lemmatizer.lemmatize(base)
            except:
                pass
        return word

    def _extract_keywords_from_text(self, text: str, keywords: List[str]) -> List[str]:
        """从文本中提取匹配的关键词"""
        if not text:
            return []

        # 合并标题和摘要进行分析
        content = text.lower()
        matched = []

        for keyword in keywords:
            normalized_keyword = self._normalize_word(keyword)
            # 使用单词边界匹配
            pattern = r'\b' + re.escape(normalized_keyword) + r'(s|ing|ed)?\b'
            if re.search(pattern, content):
                matched.append(keyword)

        return matched

    def fetch_papers(self) -> List[Dict]:
        """从ArXiv获取论文"""
        # 计算最近30天的日期范围
        end_date = datetime.utcnow()
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

        print(f"正在从ArXiv获取论文...")
        print(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

        response = requests.get(url, timeout=30)
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

        print(f"获取到 {len(papers)} 篇论文")
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

    def analyze_paper(self, paper: Dict, keywords: List[str]) -> Dict:
        """分析单篇论文"""
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

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
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

            # 尝试解析JSON
            import json
            # 移除可能的markdown代码块标记
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            analysis = json.loads(content)
            return analysis

        except Exception as e:
            print(f"分析论文时出错: {title[:50]}... - {str(e)}")
            return {
                'translated_title': f"[翻译失败] {title}",
                'company': '未知',
                'summary': f"摘要生成失败: {str(e)}",
                'interest_score': 1,
                'interest_reason': 'API调用失败'
            }


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
            authors = paper.get('authors', [])[:max_authors]
            if len(paper.get('authors', [])) > max_authors:
                authors.append(f"等{len(paper['authors'])}人")

            author_str = "、".join(authors)

            # 处理时间
            submitted_date = paper.get('submitted_date', '')
            if submitted_date:
                try:
                    dt = datetime.fromisoformat(submitted_date.replace('Z', '+00:00'))
                    date_str = dt.strftime("%Y-%m-%d")
                except:
                    date_str = submitted_date[:10]
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

        print(f"报告已保存到: {filepath}")
        return str(filepath)


def main():
    """主函数"""
    print("=" * 60)
    print("ArXiv cs.IR 论文抓取和分析工具")
    print("=" * 60)

    # 初始化
    fetcher = ArxivFetcher()
    analyzer = DeepSeekAnalyzer(fetcher.config)
    reporter = ReportGenerator(fetcher.config)

    run_time = datetime.now()

    # 1. 获取论文
    papers = fetcher.fetch_papers()
    if not papers:
        print("未获取到论文，程序退出")
        return

    # 2. 排序并取top N
    top_papers = fetcher.rank_papers(papers)
    print(f"按关键词排序后，选取Top {len(top_papers)} 篇论文")

    # 3. 使用DeepSeek分析每篇论文
    keywords = fetcher._get_keywords()
    analyzed_papers = []

    print(f"\n开始使用DeepSeek分析论文...")
    for idx, paper in enumerate(top_papers, 1):
        print(f"[{idx}/{len(top_papers)}] 正在分析: {paper['title'][:60]}...")
        analysis = analyzer.analyze_paper(paper, keywords)

        # 合并数据
        paper.update(analysis)
        analyzed_papers.append(paper)

    # 4. 生成报告
    print(f"\n生成报告...")
    report_content = reporter.generate(analyzed_papers, run_time)

    # 5. 保存报告
    reporter.save_report(report_content, run_time)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
