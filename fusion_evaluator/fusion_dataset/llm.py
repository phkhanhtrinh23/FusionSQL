from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    stop: Optional[List[str]] = None


class PromptLoader:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.data: Dict[str, Any] = yaml.safe_load(f)

    def get(self, key: str) -> Any:
        return self.data.get(key)


def render_template(tpl: str, **kwargs: Any) -> str:
    return tpl.format(**kwargs)


class HuggingFaceLLM:
    def __init__(self, model_name: str, device: Optional[str] = None, torch_dtype: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        dtype = None
        if torch_dtype:
            low = torch_dtype.lower()
            if low in ("fp16", "float16"):
                dtype = torch.float16
            elif low in ("bf16", "bfloat16"):
                dtype = torch.bfloat16
            elif low in ("fp32", "float32"):
                dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompts: Iterable[str], gen: GenerationConfig) -> List[str]:
        outputs: List[str] = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=gen.max_new_tokens,
                    temperature=gen.temperature,
                    top_p=gen.top_p,
                    do_sample=gen.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # Extract generated continuation
            gen_text = text[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)) :]
            if gen.stop:
                for s in gen.stop:
                    if s in gen_text:
                        gen_text = gen_text.split(s, 1)[0]
                        break
            outputs.append(gen_text.strip())
        return outputs


