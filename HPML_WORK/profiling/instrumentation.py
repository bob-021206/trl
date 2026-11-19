"""Reversible instance hooks for the pinned TRL API; lifecycle behavior is not modified."""

from contextlib import contextmanager
from functools import wraps


def validate_lifecycle(generation, variant):
    # This is a capability check, not proof of correctness. Inspect recorded counts as well.
    awake_api = callable(getattr(generation, "sleep", None))
    if (not variant.startswith("baseline")) != awake_api:
        raise RuntimeError(
            f"Variant {variant!r} does not match the source lifecycle. "
            "Use affected source for baseline; integrate Binyan's patch for awake/awake_cache."
        )


@contextmanager
def instrument(trainer, recorder):
    restored = []

    def hook(obj, attr, name, before=None, after=None):
        original = getattr(obj, attr)  # Fail loudly if a future TRL revision changes the API.
        had_instance_attr = attr in vars(obj)

        @wraps(original)
        def wrapped(*args, **kwargs):
            fields = before(args, kwargs) if before else {}
            with recorder.span(name, **fields) as measured:
                result = original(*args, **kwargs)
                if after:
                    measured.update(after(result))
                return result

        setattr(obj, attr, wrapped)
        restored.append((obj, attr, original, had_instance_attr))

    rollout_number = 0
    original_score = trainer._generate_and_score_completions

    def score(*args, **kwargs):
        nonlocal rollout_number
        previous_rollout, previous_turn = recorder.rollout, recorder.turn
        recorder.rollout, recorder.turn = rollout_number, None
        rollout_number += 1
        try:
            with recorder.span("rollout_and_score"):
                return original_score(*args, **kwargs)
        finally:
            recorder.rollout, recorder.turn = previous_rollout, previous_turn

    try:
        had_score = "_generate_and_score_completions" in vars(trainer)
        trainer._generate_and_score_completions = score
        restored.append((trainer, "_generate_and_score_completions", original_score, had_score))
        hook(trainer, "_generate", "rollout", after=lambda result: {"trajectories": len(result[0])})
        hook(trainer, "_calculate_rewards", "reward")
        hook(trainer, "_get_per_token_logps_and_entropies", "policy_logprob")
        hook(trainer, "training_step", "train_microstep")
        if hasattr(trainer, "compute_loss"):
            hook(trainer, "compute_loss", "loss_forward")
        if hasattr(trainer, "prediction_step"):
            hook(trainer, "prediction_step", "eval_batch")
        if hasattr(trainer, "accelerator"):
            hook(trainer.accelerator, "backward", "backward")

        def generation_input(args, kwargs):
            recorder.turn = 0 if recorder.turn is None else recorder.turn + 1
            prompts = kwargs.get("prompts", args[0] if args else [])
            return {"batch_size": len(prompts), "prompt_tokens": sum(map(len, prompts))}

        def generation_output(result):
            return {"generated_tokens": sum(map(len, result[1]))}

        generation = trainer.vllm_generation
        hook(generation, "generate", "generation_turn", generation_input, generation_output)
        hook(generation, "sync_weights", "sync_weights")
        hook(generation.llm, "generate", "engine_generate")
        hook(generation.llm, "sleep", "engine_sleep", lambda a, k: {"level": k.get("level", a[0] if a else None)})
        hook(generation.llm, "wake_up", "engine_wake", lambda a, k: {"tags": k.get("tags")})
        hook(generation.llm, "reset_prefix_cache", "cache_reset")
        yield
    finally:
        for obj, attr, original, had_instance_attr in reversed(restored):
            if had_instance_attr:
                setattr(obj, attr, original)
            else:
                delattr(obj, attr)
