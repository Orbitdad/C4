from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


DEFAULT_CONFIG_PATH = Path("config") / "config.yaml"
EXAMPLE_CONFIG_PATH = Path("config") / "config.example.yaml"


class ConfigError(Exception):
    pass


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """
    Load YAML configuration, falling back to example config if needed.
    Environment variables (e.g. GEMINI_API_KEY) take precedence.
    """
    load_dotenv()

    config_path = path
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config: Dict[str, Any] = {}
    to_try: list[Path] = []
    if config_path and config_path.is_file():
        to_try = [config_path]
    elif config_path:
        example_in_dir = config_path.parent / "config.example.yaml"
        if example_in_dir.is_file():
            to_try = [example_in_dir]
    if not to_try and EXAMPLE_CONFIG_PATH.is_file():
        to_try = [EXAMPLE_CONFIG_PATH]

    if to_try:
        with to_try[0].open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        raise ConfigError(
            "No config file found. Create config.yaml from config.example.yaml."
        )

    # Inject environment-based overrides
    api_config = config.setdefault("llm", {})
    env_api_key = os.getenv("GEMINI_API_KEY") or os.getenv(
        api_config.get("api_key_env", "")
    )
    if env_api_key:
        api_config["api_key"] = env_api_key

    return config


def get_llm_api_key(config: Dict[str, Any]) -> str | None:
    llm_cfg = config.get("llm", {})
    return llm_cfg.get("api_key") or os.getenv(
        llm_cfg.get("api_key_env", "GEMINI_API_KEY")
    )

