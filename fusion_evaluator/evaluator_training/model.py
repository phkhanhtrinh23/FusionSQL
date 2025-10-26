from dataclasses import dataclass
from typing import Tuple

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor


@dataclass
class EvaluatorModelConfig:
	hidden_sizes: Tuple[int, ...] = (128, 64)
	seed: int = 0


class EvaluatorModel:
	def __init__(self, config: EvaluatorModelConfig = EvaluatorModelConfig()):
		self.config = config
		self.model = MLPRegressor(hidden_layer_sizes=config.hidden_sizes, random_state=config.seed, max_iter=1000)

	def fit(self, X: np.ndarray, y: np.ndarray) -> None:
		self.model.fit(X, y)

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self.model.predict(X)

	def save(self, path: str) -> None:
		joblib.dump({"config": self.config, "model": self.model}, path)

	@staticmethod
	def load(path: str) -> "EvaluatorModel":
		obj = joblib.load(path)
		inst = EvaluatorModel(obj["config"])  # type: ignore[arg-type]
		inst.model = obj["model"]
		return inst
