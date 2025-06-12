import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file.

    Uses ``yaml.safe_load`` to fully support standard YAML syntax.
    """
    config_path = Path(path)
    with config_path.open("r") as f:
        return yaml.safe_load(f)
