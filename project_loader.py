import json, re
from pathlib import Path
from typing import List, Tuple, Union


def load_project(
    path: Union[str, Path]
) -> Tuple[List[int], List[Tuple[int, int]], List[str]]:
    """
    Load a project definition from JSON (durations, precedences, names).

    Parameters
    ----------
    path : str or pathlib.Path
        File location.

    Returns
    -------
    durations   : list[int]
    precedences : list[tuple[int, int]]
    names       : list[str]
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    durations   = data["durations"]
    precedences = [tuple(p) for p in data["precedences"]]
    names       = data.get("names",
                  [f"Act {i}" for i in range(len(durations))])

    return durations, precedences, names
