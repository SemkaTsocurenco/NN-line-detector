from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigManager:
    """Loads and persists application configuration from a YAML file."""

    def __init__(self, path: Path | str = Path("config/config.yaml")) -> None:
        self.path = Path(path)
        self._config: Dict[str, Any] = {}
        self.load()

    @property
    def data(self) -> Dict[str, Any]:
        return self._config

    def load(self) -> None:
        if not self.path.exists():
            logging.warning("Config file %s not found, using empty config", self.path)
            self._config = {}
            return
        with self.path.open("r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self._config, f, sort_keys=False, allow_unicode=False)

    def get_section(self, name: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
        section = self._config.get(name, default or {})
        return copy.deepcopy(section)

    def update_section(self, name: str, updates: Dict[str, Any]) -> None:
        base = self._config.get(name, {})
        base.update(updates)
        self._config[name] = base

    def set_value(self, path: str, value: Any) -> None:
        """
        Sets a nested value using dot-notation (e.g., 'rtsp.uri').
        Creates intermediate dicts if necessary.
        """
        parts = path.split(".")
        node = self._config
        for key in parts[:-1]:
            node = node.setdefault(key, {})
        node[parts[-1]] = value

    def get_value(self, path: str, default: Any = None) -> Any:
        parts = path.split(".")
        node: Any = self._config
        for key in parts:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return copy.deepcopy(node)
