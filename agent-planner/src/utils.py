import io
import json
import os
import pickle
from pathlib import Path
from pprint import pprint
from typing import Any, Dict

import pandas as pd
import yaml
from loguru import logger


def load(path) -> Any:
    extension = path.split(".")[-1]
    try:
        if extension == "txt":
            with open(path, "r", encoding="utf-8") as f:
                loaded_file = f.read()
        elif extension == "csv":
            with open(path, "r", encoding="utf-8") as f:
                loaded_file = pd.read_csv(f, encoding="utf-8")
        elif extension == "json":
            with open(path, "r", encoding="utf-8") as f:
                loaded_file = json.load(f)
        elif extension == "yaml":
            with open(path, "r", encoding="utf-8") as f:
                loaded_file = yaml.safe_load(f)
        elif extension == "pkl":
            with open(path, "rb") as f:
                loaded_file = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

        return loaded_file
    except Exception as e:
        logger.exception("Error loading file from {}", path)
        raise FileExistsError(f"Error loading file from {path}: {e}") from e


def save(data: Any, path: str):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # print(f"Saving file to: {path}")
    sub_folder = os.path.basename(path)
    extension = sub_folder.split(".")[-1]
    try:
        if extension == "txt":
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(data)
        elif extension == "csv":
            if isinstance(data, pd.DataFrame):
                with open(path, "w", encoding="utf-8-sig") as f:
                    data.to_csv(f, index=False, encoding="utf-8-sig")
            else:
                raise ValueError("Data must be a pandas DataFrame for CSV format.")
        elif extension == "json":
            with open(path, "w", encoding="utf-8-sig") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        elif extension == "yaml":
            with open(path, "w", encoding="utf-8-sig") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        elif extension == "pkl":
            with open(path, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
    except Exception as e:
        logger.exception("Error saving file to {}", path)
