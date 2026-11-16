"""W1 chain arithmetic: environment, reward functions and the reference trace.

Use with GRPOTrainer:

    GRPOTrainer(
        ...,
        environment_factory=ChainArithmeticEnv,
        reward_funcs=[answer_reward, steps_followed_reward],
        args=GRPOConfig(reward_weights=[1.0, 0.0], ...),  # steps_followed is logged, not optimized
    )
"""

import re


class ChainArithmeticEnv:
    def reset(self, op_ids, observations, **kwargs) -> None:
        self.op_ids = op_ids
        self.observations = observations
        self.steps_followed = 0

    def get_op(self, op_id: str) -> str:
        """
        Get the operation for one step of the chain.

        Args:
            op_id: The id of the step, given in the question or in the previous result.

        Returns:
            The operation to apply and the id of the next step.
        """
        if op_id not in self.op_ids:
            return f"error: unknown op_id {op_id}. Use the op_id from the question or the previous next_id."
        i = self.op_ids.index(op_id)
        if i == self.steps_followed:
            self.steps_followed += 1
        return self.observations[i]


def answer_reward(completions, answer, **kwargs):
    rewards = []
    for completion, expected in zip(completions, answer, strict=True):
        match = re.search(r"<answer>\s*(-?\d+)\s*</answer>", completion[-1]["content"])
        rewards.append(float(match is not None and int(match.group(1)) == expected))
    return rewards


def steps_followed_reward(environments, K, **kwargs):
    """Fraction of the chain followed with the right ids. Logged with weight 0: its mean is the full-run rate proxy."""
    return [environment.steps_followed / k for environment, k in zip(environments, K, strict=True)]


def reference_trace(x0, ops, op_ids, observations, **kwargs):
    """Completion messages of a policy that follows the chain perfectly (for replay checks, budgets and SFT warmup)."""
    messages = []
    steps = [
        f"{x0 + sum(ops[:i])} {'+' if op > 0 else '-'} {abs(op)} = {x0 + sum(ops[: i + 1])}"
        for i, op in enumerate(ops)
    ]
    for i, op_id in enumerate(op_ids):
        content = f"Current value: {x0}." if i == 0 else steps[i - 1]
        call = {"type": "function", "function": {"name": "get_op", "arguments": {"op_id": op_id}}}
        messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
        messages.append({"role": "tool", "name": "get_op", "content": observations[i]})
    final = f"{steps[-1]}\n<answer>{x0 + sum(ops)}</answer>"
    messages.append({"role": "assistant", "content": final})
    return messages
