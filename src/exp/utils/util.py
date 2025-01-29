import json

import numpy as np

from exp.options import Func


def serialize_json(res, fl, idf):
    import json
    from pathlib import Path

    Path(fl).mkdir(parents=True, exist_ok=True)
    st = json.dumps(res, cls=NpEncoder, default=default)
    with open(fl + idf + ".json", "w") as fid:
        fid.write(st)


def deserialize_json(fl, idf) -> dict:
    with open(fl + idf + ".json", "r") as fid:
        d = json.load(fid)
    return d


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        elif isinstance(obj, Func):
            return str(obj)
        return super(NpEncoder, self).default(obj)


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError("Unknown type:", type(obj))
