from typing import Dict, List, Tuple

import numpy as np


def frechet_like(source: np.ndarray, target: np.ndarray) -> np.ndarray:
	# Expect arrays of shape (n, D) and (m, D)
	mu_s = source.mean(axis=0)
	mu_t = target.mean(axis=0)
	s2_s = source.var(axis=0) + 1e-8
	s2_t = target.var(axis=0) + 1e-8
	trans = np.sum((mu_t - mu_s) ** 2)
	scale = np.sum((np.sqrt(s2_t / s2_s) - 1.0) ** 2)
	return np.array([trans, scale], dtype=np.float32)


def mahalanobis_diag(source: np.ndarray, target: np.ndarray) -> np.ndarray:
	mu_s = source.mean(axis=0)
	s2_s = source.var(axis=0) + 1e-8
	inv_std = 1.0 / np.sqrt(s2_s)
	diff = (target - mu_s) * inv_std
	r = np.sqrt(np.sum(diff ** 2, axis=1))
	return np.array([r.mean(), r.std()], dtype=np.float32)


def sliced_wasserstein(source: np.ndarray, target: np.ndarray, L: int = 64, seed: int = 0) -> np.ndarray:
	rng = np.random.RandomState(seed)
	D = source.shape[1]
	vals: List[float] = []
	for _ in range(L):
		v = rng.normal(size=(D,)).astype(np.float32)
		v /= np.linalg.norm(v) + 1e-8
		s_proj = np.sort(source @ v)
		t_proj = np.sort(target @ v)
		m = min(len(s_proj), len(t_proj))
		if m == 0:
			vals.append(0.0)
			continue
		vals.append(np.sqrt(np.mean((t_proj[:m] - s_proj[:m]) ** 2)))
	return np.array([np.mean(vals)], dtype=np.float32)


def compute_delta(source: np.ndarray, target: np.ndarray, L: int = 64) -> np.ndarray:
	g = frechet_like(source, target)
	m = mahalanobis_diag(source, target)
	s = sliced_wasserstein(source, target, L=L)
	return np.concatenate([g, m, s], axis=0)
