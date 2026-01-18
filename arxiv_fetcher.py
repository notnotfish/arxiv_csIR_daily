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
import requests
from datetime import datetime, timedelta
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
            # 对关键词本身和其词形变化都进行匹配
            normalized_keyword = self._normalize_word(keyword)

            # 构建更灵活的正则：匹配原词及其常见变形
            # 例如：rank 会匹配 rank, ranks, ranking, ranked
            variants = [
                normalized_keyword,
                normalized_keyword + 's',
                normalized_keyword + 'ing',
                normalized_keyword + 'ed',
            ]

            # 如果原关键词本身就包含词形变化，也加入匹配
            if keyword.lower() != normalized_keyword:
                variants.append(keyword.lower())

            # 使用单词边界确保精确匹配
            for variant in variants:
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, content):
                    matched.append(keyword)
                    break  # 找到一个变体就够了，避免重复

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

        logger.info(f"正在从ArXiv获取论文...")
        logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

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

        # 尝试提取JSON对象（查找最外层的花括号）
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
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
        analyzed_papers = []

        logger.info(f"开始使用DeepSeek分析论文（并发数: {max_workers}）...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(self.analyze_paper, paper, keywords): paper
                for paper in papers
            }

            # 收集结果
            for idx, future in enumerate(as_completed(future_to_paper), 1):
                paper = future_to_paper[future]
                try:
                    analysis = future.result()
                    paper.update(analysis)
                    analyzed_papers.append(paper)
                    logger.info(f"[{idx}/{len(papers)}] 完成分析: {paper['title'][:60]}...")
                except Exception as e:
                    logger.error(f"并发分析出错: {paper['title'][:50]}... - {e}")
                    # 添加失败的默认结果
                    paper.update(self._get_fallback_analysis(paper, str(e)))
                    analyzed_papers.append(paper)

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

        logger.info(f"报告已保存到: {filepath}")
        return str(filepath)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("ArXiv cs.IR 论文抓取和分析工具")
    logger.info("=" * 60)

    try:
        # 初始化
        fetcher = ArxivFetcher()
        analyzer = DeepSeekAnalyzer(fetcher.config)
        reporter = ReportGenerator(fetcher.config)

        run_time = datetime.now()

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

        # 5. 保存报告
        reporter.save_report(report_content, run_time)

        logger.info("=" * 60)
        logger.info("完成！")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
