"""Shared acceptance and KV accounting for simulated verification rounds."""

from functools import wraps

import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.manager import ConfigManager

SIMULATED_SPEC_ALGORITHMS = ("STANDALONE", "DFLASH")


def acceptance_config(algorithm, max_accepted):
    config = ConfigManager.get_speculative_config(algorithm)
    default = config.get("accepted_draft_tokens", 0)
    traces = config.get("acceptance_by_request", {})
    if type(default) is not int or not 0 <= default <= max_accepted:
        raise ValueError(
            f"{algorithm}.accepted_draft_tokens must be in [0, {max_accepted}]"
        )
    if not isinstance(traces, dict) or any(
        not isinstance(trace, list)
        or any(type(n) is not int or not 0 <= n <= max_accepted for n in trace)
        for trace in traces.values()
    ):
        raise ValueError(
            f"{algorithm}.acceptance_by_request must map request IDs to lists "
            f"of integers in [0, {max_accepted}]"
        )
    return default, traces


def _assign_cpu_kv_slots(req_indices, req_to_token, starts, ends, slots, _width):
    offset = 0
    for index, start, end in zip(req_indices.tolist(), starts.tolist(), ends.tolist()):
        length = end - start
        req_to_token[index, start:end] = slots[offset : offset + length]
        offset += length


class C_SpeculativeKVPoolHook(BaseHook):
    REQUIRED = False
    HOOK_CLASS_NAME = "MHATokenToKVPool"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.memory_pool"

    @classmethod
    def hook(cls, target):
        original_warmup = getattr(target, "_init_kv_copy_and_warmup", None)
        if original_warmup is None:
            return

        @wraps(original_warmup)
        def warmup(self):
            from sglang.srt.runtime_context import get_spec

            if (
                torch.device(self.device).type == "cpu"
                and get_spec().speculative_algorithm in SIMULATED_SPEC_ALGORITHMS
            ):
                # The simulated worker does not copy numeric KV payloads.
                self._kv_copy_config = None
                return
            return original_warmup(self)

        target._init_kv_copy_and_warmup = warmup


def install_cpu_kv_slot_fallback():
    # Preserve native allocation with CUDA-only kernel wheels.
    if not hasattr(torch.ops.sgl_kernel, "assign_req_to_token_pool_cpu"):
        from sglang.srt.mem_cache import allocation

        allocation.assign_req_to_token_pool_cpu = _assign_cpu_kv_slots


def simulate_speculative_batch(worker, batch):
    from sglang.srt.managers.utils import GenerationBatchResult

    if batch.forward_mode.is_extend():
        result = worker.target_worker.forward_batch_generation(batch)
        result.new_seq_lens = batch.seq_lens
        return result, result.next_token_ids.to(torch.int32)

    default, traces = worker._simulated_acceptance
    commits = []
    for req in batch.reqs:
        trace = traces.get(req.rid, ())
        step = req.spec_verify_ct
        commits.append(1 + (trace[step] if step < len(trace) else default))
    accept_lens = torch.tensor(commits, dtype=torch.int32, device=worker.device)
    bonus_tokens = torch.ones(len(commits), dtype=torch.int32, device=worker.device)
    width = worker.speculative_num_draft_tokens
    result = GenerationBatchResult(
        next_token_ids=bonus_tokens.repeat_interleave(width),
        accept_lens=accept_lens,
        speculative_num_draft_tokens=width,
        new_seq_lens=batch.seq_lens + accept_lens,
    )
    return result, bonus_tokens
