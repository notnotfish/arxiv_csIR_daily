"""报告生成器 - 生成Markdown和HTML报告"""
import logging
import markdown
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Markdown和HTML报告生成器"""

    def __init__(self, config: Dict, target_timezone: timezone):
        """
        初始化报告生成器

        Args:
            config: 配置字典
            target_timezone: 目标时区
        """
        self.config = config
        self.timezone = target_timezone

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

        markdown_content = f"""# ArXiv cs.IR 每日论文摘要

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
                    # arXiv返回的是UTC时间，需要转换为配置的时区
                    dt = datetime.fromisoformat(submitted_date.replace('Z', '+00:00'))
                    # 转换为配置的时区
                    dt = dt.astimezone(self.timezone)
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
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

            markdown_content += f"""## {idx}. {paper.get('translated_title', paper['title'])}

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

        return markdown_content

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

        # 鸭子SVG favicon
        duck_favicon = '''<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,
        <svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22>
            <ellipse cx=%2250%22 cy=%2275%22 rx=%2235%22 ry=%2220%22 fill=%22%23FFD700%22 stroke=%22%23E6C200%22 stroke-width=%223%22/>
            <ellipse cx=%2250%22 cy=%2270%22 rx=%2225%22 ry=%2215%22 fill=%22%23FFF8E7%22/>
            <circle cx=%2240%22 cy=%2260%22 r=%228%22 fill=%22%23333%22/>
            <circle cx=%2260%22 cy=%2260%22 r=%228%22 fill=%22%23333%22/>
            <circle cx=%2242%22 cy=%2258%22 r=%223%22 fill=%22%23FFF%22/>
            <circle cx=%2262%22 cy=%2258%22 r=%223%22 fill=%22%23FFF%22/>
            <ellipse cx=%2250%22 cy=%2270%22 rx=%2210%22 ry=%226%22 fill=%22%23FF6B35%22/>
            <path d=%22Q 35 72, 25 85 Q 20 95, 35 95 Q 50 95, 45 85%22 fill=%22%23FF8C00%22 stroke=%22%23E67300%22 stroke-width=%222%22/>
            <path d=%22Q 65 72, 75 85 Q 80 95, 65 95 Q 50 95, 55 85%22 fill=%22%23FF8C00%22 stroke=%22%23E67300%22 stroke-width=%222%22/>
            <ellipse cx=%2250%22 cy=%2220%22 rx=%2215%22 ry=%2212%22 fill=%22%238B4513%22/>
        </svg>">'''

        # HTML模板
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArXiv cs.IR 每日论文摘要 - {date_str}</title>
    {duck_favicon}
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
        """
        初始化网页生成器

        Args:
            config: 配置字典
        """
        self.config = config
        self.output_dir = Path(config['report']['output_dir'])

    def generate_index(self) -> str:
        """生成主页 index.html"""
        reports = self._scan_reports()

        # 获取最新报告
        latest_report = reports[0] if reports else None

        # 最近10条报告
        recent_reports = reports[:10]

        # 鸭子SVG favicon
        duck_favicon = '''<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,
        <svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22>
            <ellipse cx=%2250%22 cy=%2275%22 rx=%2235%22 ry=%2220%22 fill=%22%23FFD700%22 stroke=%22%23E6C200%22 stroke-width=%223%22/>
            <ellipse cx=%2250%22 cy=%2270%22 rx=%2225%22 ry=%2215%22 fill=%22%23FFF8E7%22/>
            <circle cx=%2240%22 cy=%2260%22 r=%228%22 fill=%22%23333%22/>
            <circle cx=%2260%22 cy=%2260%22 r=%228%22 fill=%22%23333%22/>
            <circle cx=%2242%22 cy=%2258%22 r=%223%22 fill=%22%23FFF%22/>
            <circle cx=%2262%22 cy=%2258%22 r=%223%22 fill=%22%23FFF%22/>
            <ellipse cx=%2250%22 cy=%2270%22 rx=%2210%22 ry=%226%22 fill=%22%23FF6B35%22/>
            <path d=%22Q 35 72, 25 85 Q 20 95, 35 95 Q 50 95, 45 85%22 fill=%22%23FF8C00%22 stroke=%22%23E67300%22 stroke-width=%222%22/>
            <path d=%22Q 65 72, 75 85 Q 80 95, 65 95 Q 50 95, 55 85%22 fill=%22%23FF8C00%22 stroke=%22%23E67300%22 stroke-width=%222%22/>
            <ellipse cx=%2250%22 cy=%2220%22 rx=%2215%22 ry=%2212%22 fill=%22%238B4513%22/>
        </svg>">'''

        # 生成HTML（使用普通字符串，不需要转义花括号）
        html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArXiv cs.IR 每日论文摘要</title>
    {duck_favicon}
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

        html = html_template.replace('{duck_favicon}', duck_favicon)

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
