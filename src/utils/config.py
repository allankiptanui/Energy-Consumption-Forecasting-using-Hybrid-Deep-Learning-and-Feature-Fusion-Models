#  YAML config loader.
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
 
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found at {path}. Create one or check the path.")
    try:
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML config at {path}: {e}") from e
    # If file is empty, yaml.safe_load returns None
    if cfg is None:
        cfg = {}
    return cfg


def get_cfg_value(cfg: Dict[str, Any], keys: List[str], default: Optional[Any] = None) -> Any:
 
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node

