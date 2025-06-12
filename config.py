import json
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    """Load configuration from a YAML/JSON file.

    The parser expects the file to contain JSON-formatted data, which is a
    subset of YAML. This avoids the need for external YAML libraries.
    """
    config_path = Path(path)
    with config_path.open("r") as f:
        return json.load(f)
