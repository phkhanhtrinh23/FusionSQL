from fusion_evaluator.metrics.fusion import fusion_score, FusionWeights


def test_fusion_score_bounds():
	w = FusionWeights()
	assert 0.0 <= fusion_score(0.0, 0.0, 0.0, w) <= 1.0
	assert 0.0 <= fusion_score(1.0, 1.0, 1.0, w) <= 1.0
