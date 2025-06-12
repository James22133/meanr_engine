import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from a YAML/JSON file.

    Uses ``yaml.safe_load`` so standard YAML features like comments are
    supported.
    """
    config_path = Path(path)
    with config_path.open("r") as f:
        return yaml.safe_load(f)
