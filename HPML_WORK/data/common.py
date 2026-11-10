"""Token-length measurement shared by the W1/W2 generators, checks and environments.

Every length knob (P, L_obs, L_doc) is measured with the Qwen3 chat template, exactly the way GRPOTrainer
concatenates token ids during a multi-turn rollout, so the numbers in the datasets are the numbers vLLM sees.
"""

import hashlib
import json

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.utils import get_json_schema


WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


class Lengths:
    """Measure token lengths under the Qwen3 chat template with `enable_thinking=False`."""

    def __init__(self, tokenizer_name, tools):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tools = [get_json_schema(tool) for tool in tools]
        self.tool_name = self.tools[0]["function"]["name"]
        # Same dummy conversation as `GRPOTrainer._get_tool_suffix_ids`
        self._dummy = [
            {"role": "user", "content": "dummy"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": self.tool_name, "arguments": {}}}],
            },
        ]
        prefix_ids = self._apply(self._dummy, add_generation_prompt=False)
        eos = self.tokenizer.eos_token_id
        self._dummy_len = max(i for i, tok in enumerate(prefix_ids) if tok == eos) + 1

    def _apply(self, messages, add_generation_prompt, tools=None):
        return self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
            enable_thinking=False,
        )

    def tool_suffix(self, content):
        """Number of tokens GRPOTrainer appends for one tool result (includes the next generation prompt)."""
        tool_message = {"role": "tool", "name": self.tool_name, "content": content}
        return len(self._apply(self._dummy + [tool_message], add_generation_prompt=True)) - self._dummy_len

    def system(self, system_content):
        """Number of tokens of the rendered system block (system text + tool schema), up to its `<|im_end|>`."""
        ids = self._apply([{"role": "system", "content": system_content}], False, tools=self.tools)
        return ids.index(self.tokenizer.eos_token_id) + 1

    def prompt(self, prompt):
        """Number of prompt tokens fed to vLLM for the first turn."""
        return len(self._apply(prompt, add_generation_prompt=True, tools=self.tools))

    def text(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generated(self, content, arguments=None):
        """Number of tokens the model generates for one assistant turn (Qwen3 tool-call format)."""
        if arguments is not None:
            call = json.dumps({"name": self.tool_name, "arguments": arguments}, ensure_ascii=False)
            content = f"{content}\n<tool_call>\n{call}\n</tool_call>"
        return self.text(content + "<|im_end|>")

    def fit(self, measure, head, corpus_ids, start, target):
        """
        Return `head + decode(corpus_ids[start:start + n])` such that `measure(text) == target`.

        Decoding then re-encoding does not always round-trip, so `n` is corrected until the measured length is exact.
        Returns `None` when the boundary merges never land on `target`; the caller then picks another `start`.
        """
        n = target - measure(head)
        if n < 0:
            raise ValueError(f"The fixed part is already {measure(head)} tokens, longer than the target {target}.")
        seen = set()
        while n >= 0 and n not in seen:
            seen.add(n)
            text = head + self.tokenizer.decode(corpus_ids[start : start + n])
            length = measure(text)
            if length == target:
                return text
            n += target - length
        return None


def load_corpus_ids(tokenizer):
    """Token ids of wikitext-2 (train split, headings removed), used as neutral filler text."""
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", revision=WIKITEXT_REVISION)
    lines = [line for line in dataset["text"] if line.strip() and not line.startswith(" =")]
    text = "".join(lines).replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def fit_from_corpus(lengths, measure, head, corpus_ids, rng, target):
    """Fit `head` plus a random corpus window to exactly `target` tokens, retrying other windows if needed."""
    while True:
        start = rng.randrange(len(corpus_ids) - target - 64)
        text = lengths.fit(measure, head, corpus_ids, start, target)
        if text is not None:
            return text


def trace_completion_len(lengths, messages):
    """Completion tokens of a trace (assistant/tool messages), counted the way GRPOTrainer concatenates them."""
    total = 0
    for message in messages:
        if message["role"] == "tool":
            total += lengths.tool_suffix(message["content"])
        else:
            arguments = message["tool_calls"][0]["function"]["arguments"] if "tool_calls" in message else None
            total += lengths.generated(message["content"], arguments)
    return total


def round_up(n, multiple=64):
    return -(-n // multiple) * multiple


def rows_sha256(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
