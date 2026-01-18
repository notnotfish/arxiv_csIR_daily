"""工具函数和配置加载"""
import os
import re
import yaml
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any


class TimezoneFormatter(logging.Formatter):
    """支持自定义时区的日志格式化器"""

    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = tz or timezone.utc

    def formatTime(self, record, datefmt=None):
        """覆盖formatTime方法以使用自定义时区"""
        dt = datetime.fromtimestamp(record.created, tz=self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime('%Y-%m-%d %H:%M:%S')


def setup_logging(tz: timezone = None):
    """配置日志系统，使用指定时区"""
    if tz is None:
        tz = timezone(timedelta(hours=8))  # 默认UTC+8

    # 创建handler
    handler = logging.StreamHandler()

    # 创建自定义格式化器
    formatter = TimezoneFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        tz=tz
    )
    handler.setFormatter(formatter)

    # 配置root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除现有handlers
    logger.handlers.clear()
    logger.addHandler(handler)


def load_config(config_path: str) -> Dict:
    """加载配置文件，支持本地覆盖和环境变量替换"""
    # 加载主配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 尝试加载本地覆盖配置
    local_config_path = config_path.replace('.yaml', '.local.yaml')
    if os.path.exists(local_config_path):
        logging.info(f"检测到本地配置文件: {local_config_path}")
        with open(local_config_path, 'r', encoding='utf-8') as f:
            local_config = yaml.safe_load(f)
            if local_config:
                _deep_merge(config, local_config)

    # 替换环境变量
    config = _replace_env_vars(config)

    return config


def _deep_merge(base: Dict, override: Dict) -> None:
    """深度合并字典"""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _replace_env_vars(obj: Any) -> Any:
    """递归替换配置中的环境变量"""
    if isinstance(obj, dict):
        return {k: _replace_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # 匹配 ${VAR_NAME} 或 $VAR_NAME 格式
        pattern = r'\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)'
        def replace_match(match):
            var_name = match.group(1) or match.group(2)
            return os.environ.get(var_name, match.group(0))
        return re.sub(pattern, replace_match, obj)
    else:
        return obj


def validate_config(config: Dict) -> None:
    """验证配置完整性"""
    logger = logging.getLogger(__name__)

    required_fields = {
        'arxiv': ['base_url', 'category', 'max_results'],
        'deepseek': ['api_key', 'base_url', 'model'],
        'keywords': ['company', 'technical'],
        'report': ['top_n', 'max_authors', 'output_dir']
    }

    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"配置缺少必要的section: {section}")
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"配置缺少必要的字段: {section}.{field}")

    # 验证 API Key
    api_key = config['deepseek']['api_key']
    if not api_key or api_key.startswith('$'):
        raise ValueError(
            "DeepSeek API Key 未设置！\n"
            "请通过以下方式之一设置：\n"
            "1. 设置环境变量: export DEEPSEEK_API_KEY='your-key'\n"
            "2. 创建 config.local.yaml 并设置 deepseek.api_key"
        )

    logger.info("配置验证通过")


def get_timezone_from_config(config: Dict) -> timezone:
    """从配置获取时区对象"""
    tz_config = config.get('timezone', {})
    offset_hours = tz_config.get('offset_hours', 8)  # 默认UTC+8
    return timezone(timedelta(hours=offset_hours))
