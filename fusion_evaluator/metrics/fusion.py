from dataclasses import dataclass


@dataclass
class FusionWeights:
	execution: float = 0.5
	component: float = 0.4
	exact: float = 0.1


def fusion_score(execution_acc: float, component_f1: float, exact_match: float, weights: FusionWeights = FusionWeights()) -> float:
	w_sum = max(weights.execution + weights.component + weights.exact, 1e-9)
	score = (
		weights.execution * execution_acc
		+ weights.component * component_f1
		+ weights.exact * exact_match
	) / w_sum
	return score
