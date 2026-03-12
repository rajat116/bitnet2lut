"""Shared utility functions for bitnet2lut."""

import json
import logging
import sys
from pathlib import Path

import yaml
try:
    from rich.console import Console
    from rich.logging import RichHandler
    console = Console()
    _has_rich = True
except ImportError:
    _has_rich = False
    console = None


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with rich handler (if available) or basic handler."""
    level = logging.DEBUG if verbose else logging.INFO
    if _has_rich:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    return logging.getLogger("bitnet2lut")


def load_config(config_path: str | Path | None = None) -> dict:
    """Load pipeline configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        # Fall back to package-relative path
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict, path: str | Path) -> None:
    """Save dictionary as JSON with readable formatting."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def format_size(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def format_count(n: int) -> str:
    """Format large number with commas."""
    return f"{n:,}"
