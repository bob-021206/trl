"""W2 multi-hop retrieval: shared knowledge base, environment, reward functions and the reference trace.

Use with GRPOTrainer (the knowledge base and its BM25 index are loaded once and shared by all environments):

    kb = KnowledgeBase("HPML_WORK/data/out/w2/kb")
    GRPOTrainer(
        ...,
        environment_factory=lambda: KnowledgeSearchEnv(kb),
        reward_funcs=[answer_reward, hops_followed_reward],
        args=GRPOConfig(reward_weights=[1.0, 0.0], ...),  # hops_followed is logged, not optimized
    )
"""

import json
import re
from pathlib import Path

import bm25s


class KnowledgeBase:
    """Read-only knowledge base. Results are deterministic and cached per (query, top_k)."""

    def __init__(self, kb_dir):
        kb_dir = Path(kb_dir)
        with open(kb_dir / "entities.jsonl") as f:
            self.entities = [json.loads(line) for line in f]
        self.index = {entity["name"]: i for i, entity in enumerate(self.entities)}
        self.retriever = bm25s.BM25.load(kb_dir / "bm25")
        self._cache = {}

    def search_ids(self, query, top_k):
        if (query, top_k) not in self._cache:
            tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
            ids, _ = self.retriever.retrieve(tokens, k=top_k, show_progress=False)
            self._cache[query, top_k] = [int(i) for i in ids[0]]
        return self._cache[query, top_k]

    def search(self, query, top_k):
        ids = self.search_ids(query, top_k)
        return "\n\n".join(f"[{rank + 1}] {self.entities[i]['doc']}" for rank, i in enumerate(ids))


class KnowledgeSearchEnv:
    def __init__(self, kb):
        self.kb = kb

    def reset(self, gold_entities, top_k, **kwargs) -> None:
        self.gold_entities = gold_entities
        self.top_k = top_k
        self.hops_followed = 0

    def search(self, query: str) -> str:
        """
        Search the knowledge base for documents about an entity.

        Args:
            query: The exact name of the entity to look up.

        Returns:
            The best matching documents, numbered by rank.
        """
        if self.hops_followed < len(self.gold_entities) and query.strip() == self.gold_entities[self.hops_followed]:
            self.hops_followed += 1
        return self.kb.search(query, self.top_k)


def answer_reward(completions, answer, **kwargs):
    rewards = []
    for completion, expected in zip(completions, answer, strict=True):
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", completion[-1]["content"])
        rewards.append(float(match is not None and match.group(1) == expected))
    return rewards


def hops_followed_reward(environments, K, **kwargs):
    """Fraction of hops searched with the right entity, in order. Logged with weight 0."""
    return [environment.hops_followed / k for environment, k in zip(environments, K, strict=True)]


def reference_trace(kb, relations, attribute, gold_entities, answer, top_k, **kwargs):
    """Completion messages of a policy that follows the chain perfectly (for replay checks, budgets and SFT warmup)."""
    messages = []
    for i, entity in enumerate(gold_entities):
        if i == 0:
            content = f"Looking up {entity}."
        else:
            content = f"The {relations[i - 1].replace('_', ' ')} of {gold_entities[i - 1]} is {entity}."
        call = {"type": "function", "function": {"name": "search", "arguments": {"query": entity}}}
        messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
        messages.append({"role": "tool", "name": "search", "content": kb.search(entity, top_k)})
    final = f"The {attribute.replace('_', ' ')} of {gold_entities[-1]} is {answer}.\n<answer>{answer}</answer>"
    messages.append({"role": "assistant", "content": final})
    return messages
