"""Shared acceptance and KV accounting for simulated speculative decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from sglang_simulator.hook import BaseHook, is_class_hook_matched
from sglang_simulator.simulation.manager import ConfigManager

SIMULATED_SPEC_ALGORITHMS = ("STANDALONE", "DFLASH")


def _validate_accepted_count(
    value: Any, max_accepted: int, label: str
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= max_accepted:
        raise ValueError(f"{label} must be an integer in [0, {max_accepted}], got {value!r}")
    return int(value)


def acceptance_config(
    algorithm: str, max_accepted: int
) -> tuple[int, Mapping[str, Sequence[int]]]:
    config = ConfigManager.get_speculative_config(algorithm)
    default = _validate_accepted_count(
        config.get("accepted_draft_tokens", 0),
        max_accepted,
        f"{algorithm}.accepted_draft_tokens",
    )
    traces = config.get("acceptance_by_request") or {}
    return default, traces


def validate_speculative_hooks(spec_algorithm: Any) -> None:
    if spec_algorithm.name not in SIMULATED_SPEC_ALGORITHMS:
        return

    from sglang_simulator.simulation.sglang.dflash import C_DFlashWorkerHook
    from sglang_simulator.simulation.sglang.standalone import C_StandaloneWorkerHook

    worker_hook = {
        "STANDALONE": C_StandaloneWorkerHook,
        "DFLASH": C_DFlashWorkerHook,
    }[spec_algorithm.name]
    if not is_class_hook_matched(worker_hook):
        raise RuntimeError(
            f"{spec_algorithm.name} simulation requires "
            f"SGLang's {worker_hook.HOOK_CLASS_NAME}."
        )


class C_SpeculativeKVPoolHook(BaseHook):
    HOOK_CLASS_NAME = "MHATokenToKVPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original_init_kv_copy_and_warmup = getattr(
            target, "_init_kv_copy_and_warmup", None
        )
        if original_init_kv_copy_and_warmup is None:
            return

        def wrapped_init_kv_copy_and_warmup(self):
            from sglang.srt.runtime_context import get_spec

            if get_spec().speculative_algorithm in SIMULATED_SPEC_ALGORITHMS:
                # Simulated speculative workers do not copy numeric KV payloads.
                self._kv_copy_config = None
                return
            return original_init_kv_copy_and_warmup(self)

        target._init_kv_copy_and_warmup = wrapped_init_kv_copy_and_warmup


def simulate_speculative_batch(worker: Any, batch: Any):
    from sglang.srt.managers.utils import GenerationBatchResult

    if batch.forward_mode.is_extend():
        result = worker.target_worker.forward_batch_generation(batch)
        result.new_seq_lens = batch.seq_lens
        return result, result.next_token_ids.to(torch.int32)

    default, traces = worker._simulated_acceptance
    width = worker.speculative_num_draft_tokens
    max_accepted = width - 1
    commits = []
    for req in batch.reqs:
        trace = traces.get(req.rid, ())
        step = req.spec_verify_ct
        if step < len(trace):
            accepted = _validate_accepted_count(
                trace[step],
                max_accepted,
                f"acceptance_by_request[{req.rid!r}][{step}]",
            )
        else:
            accepted = default
        commits.append(1 + accepted)

    accept_lens = torch.tensor(commits, dtype=torch.int32, device=worker.device)
    bonus_tokens = torch.ones(len(commits), dtype=torch.int32, device=worker.device)
    result = GenerationBatchResult(
        next_token_ids=bonus_tokens.repeat_interleave(width),
        accept_lens=accept_lens,
        speculative_num_draft_tokens=width,
        new_seq_lens=batch.seq_lens + accept_lens,
    )
    return result, bonus_tokens
