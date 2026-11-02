"""集中配置：所有可变项都来自环境变量，仓库中不保存任何密钥。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Settings:
    # --- 模型 ---
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip()
    model: str = os.getenv("LLM_MODEL", "deepseek-chat").strip()
    timeout: float = _env_float("LLM_TIMEOUT", 60.0)
    max_tool_rounds: int = _env_int("MAX_TOOL_ROUNDS", 4)

    # --- 演示模式：没有 API Key 时可用固定桩模型跑通交互 ---
    mock: bool = os.getenv("LLM_MOCK", "0").strip() == "1"

    # --- 路径 ---
    db_path: Path = BASE_DIR / os.getenv("DB_PATH", "data/history.db")
    chart_dir: Path = BASE_DIR / os.getenv("CHART_DIR", "data/charts")
    knowledge_dir: Path = BASE_DIR / os.getenv("KNOWLEDGE_DIR", "knowledge")
    log_dir: Path = BASE_DIR / "logs"

    # --- 检索 ---
    retrieval_top_k: int = _env_int("RETRIEVAL_TOP_K", 3)
    retrieval_min_score: float = _env_float("RETRIEVAL_MIN_SCORE", 1.2)

    def ensure_dirs(self) -> None:
        for path in (self.db_path.parent, self.chart_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
