import json
from pathlib import Path

class Calibration:
    def __init__(self, filepath):
        with open(filepath, "r") as f:
            self.parameters = json.load(f)

    def __getattr__(self, name):
        try:
            return self.parameters[name]
        except KeyError:
            raise AttributeError(
                f"Calibration parameter '{name}' not found"
            ) from None