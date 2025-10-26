from dataclasses import dataclass
from typing import Tuple

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


@dataclass
class EvaluatorModelConfig:
	hidden_sizes: Tuple[int, ...] = (128, 64)
	seed: int = 0


class EvaluatorModel:
	def __init__(self, config: EvaluatorModelConfig = EvaluatorModelConfig()):
		self.config = config
		self.model = MLPRegressor(hidden_layer_sizes=config.hidden_sizes, random_state=config.seed, max_iter=1000)
		self.calibrator: IsotonicRegression | None = None

	def fit(self, X: np.ndarray, y: np.ndarray, calibrate: bool = True) -> None:
		self.model.fit(X, y)
		if calibrate and len(y) >= 10:
			X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=min(0.2, max(2, int(0.2 * len(y))) / len(y)), random_state=self.config.seed)
			self.model.fit(X_tr, y_tr)
			pred_cal = self.model.predict(X_cal)
			self.calibrator = IsotonicRegression(out_of_bounds="clip")
			self.calibrator.fit(pred_cal, y_cal)

	def predict(self, X: np.ndarray) -> np.ndarray:
		pred = self.model.predict(X)
		if self.calibrator is not None:
			pred = self.calibrator.predict(pred)
		return pred

	def save(self, path: str) -> None:
		joblib.dump({"config": self.config, "model": self.model, "calibrator": self.calibrator}, path)

	@staticmethod
	def load(path: str) -> "EvaluatorModel":
		obj = joblib.load(path)
		inst = EvaluatorModel(obj["config"])  # type: ignore[arg-type]
		inst.model = obj["model"]
		inst.calibrator = obj.get("calibrator")
		return inst
