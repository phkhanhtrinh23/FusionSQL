import glob
import os
from typing import Iterable, List


def discover_csvs(root_dirs: List[str]) -> List[str]:
	paths: List[str] = []
	for root in root_dirs:
		paths.extend(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
	return sorted(list(set(paths)))


def discover_sqlite_dbs(root_dirs: List[str]) -> List[str]:
	paths: List[str] = []
	for root in root_dirs:
		paths.extend(glob.glob(os.path.join(root, "**", "*.sqlite"), recursive=True))
		paths.extend(glob.glob(os.path.join(root, "**", "*.db"), recursive=True))
	return sorted(list(set(paths)))
