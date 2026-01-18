"""DeepSeek API论文分析器"""
import time
import json
import logging
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class DeepSeekAnalyzer:
    """使用DeepSeek API分析论文"""

    def __init__(self, config: Dict):
        """
        初始化分析器

        Args:
            config: 完整的配置字典
        """
        self.config = config['deepseek']
        self.api_key = self.config['api_key']
        self.base_url = self.config['base_url']
        self.model = self.config.get('model', 'deepseek-chat')
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 2)

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

    def _get_fallback_analysis(self, paper: Dict, error_msg: str) -> Dict:
        """返回失败时的默认分析结果"""
        return {
            'translated_title': f"[翻译失败] {paper['title']}",
            'company': '未知',
            'summary': f"摘要生成失败: {error_msg}",
            'interest_score': 1,
            'interest_reason': 'API调用失败'
        }
