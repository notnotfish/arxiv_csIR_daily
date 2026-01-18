"""
ArXiv论文抓取和分析工具 - 主入口
每日自动抓取cs.IR领域论文，使用DeepSeek进行翻译和摘要
"""
import os
import logging
import argparse
from datetime import datetime

from src import (
    ArxivFetcher,
    DeepSeekAnalyzer,
    ReportGenerator,
    WebGenerator,
    setup_logging,
    load_config,
    validate_config,
    get_timezone_from_config
)

# 默认日志配置（稍后会被setup_logging覆盖）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
        # 加载配置
        config = load_config(args.config)
        validate_config(config)
        target_timezone = get_timezone_from_config(config)

        # 初始化组件
        fetcher = ArxivFetcher(config, target_timezone)
        analyzer = DeepSeekAnalyzer(config)
        reporter = ReportGenerator(config, target_timezone)

        # 设置日志时区
        setup_logging(target_timezone)

        # 使用配置的时区
        run_time = datetime.now(target_timezone)

        # 1. 获取论文
        papers = fetcher.fetch_papers()
        if not papers:
            logger.warning("未获取到论文，程序退出")
            return

        # 2. 排序并取top N
        top_papers = fetcher.rank_papers(papers)
        logger.info(f"按关键词排序后，选取Top {len(top_papers)} 篇论文")

        # 3. 使用DeepSeek并发分析论文
        keywords = config['keywords'].get('company', []) + config['keywords'].get('technical', [])
        max_workers = config.get('deepseek', {}).get('max_workers', 5)
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
        web_gen = WebGenerator(config)
        web_gen.generate_index()

        logger.info("=" * 60)
        logger.info("完成！")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
