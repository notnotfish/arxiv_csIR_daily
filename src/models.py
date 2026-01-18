"""数据模型定义"""
from dataclasses import dataclass, field
from typing import List


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
