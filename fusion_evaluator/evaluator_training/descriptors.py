from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA


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


def _random_unit_directions(D: int, R: int, rng: np.random.RandomState) -> np.ndarray:
	V = rng.normal(size=(R, D)).astype(np.float32)
	V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
	return V  # (R, D)


def _pca_directions(joint: np.ndarray, k: int, seed: int) -> np.ndarray:
	if k <= 0:
		return np.zeros((0, joint.shape[1]), dtype=np.float32)
	pca = PCA(n_components=min(k, joint.shape[1]), svd_solver="randomized", random_state=seed)
	pca.fit(joint)
	# sklearn returns components_ as (k, D), each a unit direction in feature space
	return pca.components_.astype(np.float32)


def sliced_wasserstein_hybrid(
	source: np.ndarray,
	target: np.ndarray,
	k: int = 8,
	R: int = 16,
	seed: int = 0,
	subsample: Optional[int] = None,
) -> np.ndarray:
	D = source.shape[1]
	rng = np.random.RandomState(seed)
	# Subsample joint embeddings for PCA to cap cost
	joint = np.concatenate([source, target], axis=0)
	if subsample is not None and subsample > 0 and len(joint) > subsample:
		idx = rng.choice(len(joint), size=subsample, replace=False)
		joint_pca = joint[idx]
	else:
		joint_pca = joint
	V_pca = _pca_directions(joint_pca, k=k, seed=seed)  # (k, D)
	V_rand = _random_unit_directions(D, R, rng)  # (R, D)
	V = np.concatenate([V_pca, V_rand], axis=0)  # (L, D)

	# Vectorized projection: (N, D) @ (D, L) => (N, L)
	s_proj = np.sort(source @ V.T, axis=0)
	t_proj = np.sort(target @ V.T, axis=0)
	# Align by min length per direction
	m = min(s_proj.shape[0], t_proj.shape[0])
	if m == 0 or V.shape[0] == 0:
		return np.array([0.0], dtype=np.float32)
	diff = t_proj[:m, :] - s_proj[:m, :]
	col_rmse = np.sqrt(np.mean(diff ** 2, axis=0))  # (L,)
	return np.array([float(np.mean(col_rmse))], dtype=np.float32)


def compute_delta(
	source: np.ndarray,
	target: np.ndarray,
	L: int = 64,
	*,
	use_hybrid_swd: bool = False,
	pca_k: int = 8,
	rand_r: int = 16,
	subsample: Optional[int] = None,
	seed: int = 0,
) -> np.ndarray:
	g = frechet_like(source, target)
	m = mahalanobis_diag(source, target)
	if use_hybrid_swd:
		s = sliced_wasserstein_hybrid(source, target, k=pca_k, R=rand_r, seed=seed, subsample=subsample)
	else:
		s = sliced_wasserstein(source, target, L=L, seed=seed)
	return np.concatenate([g, m, s], axis=0)
