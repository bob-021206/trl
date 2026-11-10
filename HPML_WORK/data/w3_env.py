"""W3 MuSiQue: environment, reward functions and the reference trace.

Each question carries its own 20 paragraphs (supporting + distractors); `search` runs BM25 over them, so there is no
global index. Unlike W1/W2, the number of searches is up to the model: K (the annotated hop count) is the reference.

    GRPOTrainer(
        ...,
        environment_factory=MusiqueSearchEnv,
        reward_funcs=[answer_reward, supporting_found_reward],
        args=GRPOConfig(reward_weights=[1.0, 0.0], ...),  # supporting_found is logged, not optimized
    )
"""

import re
import string

import bm25s


class MusiqueSearchEnv:
    def reset(self, docs, supporting_idx, top_k, **kwargs) -> None:
        self.docs = docs
        self.supporting_idx = supporting_idx
        self.top_k = top_k
        self.retriever = bm25s.BM25()
        self.retriever.index(bm25s.tokenize(docs, stopwords="en", show_progress=False), show_progress=False)
        self.found = set()

    def search(self, query: str) -> str:
        """
        Search the paragraphs available for this question.

        Args:
            query: What to look up, for example an entity name and the fact you need.

        Returns:
            The best matching paragraphs, numbered by rank.
        """
        ids = search_ids(self.retriever, query, self.top_k)
        self.found |= set(ids) & set(self.supporting_idx)
        return "\n\n".join(f"[{rank + 1}] {self.docs[i]}" for rank, i in enumerate(ids))


def search_ids(retriever, query, top_k):
    ids, _ = retriever.retrieve(
        bm25s.tokenize([query], stopwords="en", show_progress=False), k=top_k, show_progress=False
    )
    return [int(i) for i in ids[0]]


def normalize(text):
    """SQuAD-style answer normalization."""
    text = "".join(c for c in text.lower() if c not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_reward(completions, answer, answer_aliases, **kwargs):
    rewards = []
    for completion, expected, aliases in zip(completions, answer, answer_aliases, strict=True):
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", completion[-1]["content"], re.DOTALL)
        # Lenient match: a gold answer (or alias) appears in the predicted answer, which is often a full sentence
        prediction = normalize(match.group(1)) if match else ""
        rewards.append(float(any(normalize(a) and normalize(a) in prediction for a in [expected] + aliases)))
    return rewards


def supporting_found_reward(environments, **kwargs):
    """Fraction of the supporting paragraphs the model retrieved. Logged with weight 0."""
    return [len(environment.found) / len(environment.supporting_idx) for environment in environments]


def reference_queries(decomposition, decomposition_answers):
    """One query per hop: the annotated sub-question with `#n` replaced by the answer of hop n."""
    queries = []
    for question in decomposition:
        query = re.sub(r"#(\d)", lambda m: decomposition_answers[int(m.group(1)) - 1], question)
        queries.append(query.replace(" >> ", " "))
    return queries


def reference_trace(docs, decomposition, decomposition_answers, supporting_idx, answer, top_k, **kwargs):
    """Completion messages of a policy that searches each annotated sub-question once, then answers."""
    env = MusiqueSearchEnv()
    env.reset(docs, supporting_idx, top_k)
    messages = []
    for query in reference_queries(decomposition, decomposition_answers):
        call = {"type": "function", "function": {"name": "search", "arguments": {"query": query}}}
        messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        messages.append({"role": "tool", "name": "search", "content": env.search(query)})
    messages.append({"role": "assistant", "content": f"<answer>{answer}</answer>"})
    return messages, env
