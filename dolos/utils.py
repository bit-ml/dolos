import json
import os


def cache(path, func, *args, **kwargs):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        result = func(*args, **kwargs)
        with open(path, "w") as f:
            json.dump(result, f)
        return result
