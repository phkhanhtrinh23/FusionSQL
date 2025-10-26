import random
from typing import List


STYLE_TEMPLATES = {
	"formal": "What is {ask}?",
	"colloquial": "Can you get me {ask}?",
	"imperative": "Return {ask}.",
	"interrogative": "Which {ask}?",
	"descriptive": "I need the following: {ask}.",
	"vague": "Pull details about {ask}.",
	"metaphorical": "Fish out the info about {ask}.",
	"conversational": "Hey, could you tell me {ask}?",
}

DISTRACTORS = [
	"that went viral on TikTok",
	"with a sprinkle of magic",
	"who recently trended on social",
	"that broke the internet",
]


def synthesize_question(base: str, style: str, add_distractor: bool = False) -> str:
	t = STYLE_TEMPLATES.get(style, STYLE_TEMPLATES["formal"]) 
	q = t.format(ask=base)
	if add_distractor:
		q = f"{q} {random.choice(DISTRACTORS)}"
	return q
