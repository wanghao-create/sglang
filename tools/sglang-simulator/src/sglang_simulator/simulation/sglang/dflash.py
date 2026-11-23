"""Simulate accepted drafts in the native synchronous DFlash worker."""

from contextvars import ContextVar
from functools import cache, wraps

import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.sglang.speculative import (
    acceptance_config,
    install_cpu_kv_slot_fallback,
    simulate_speculative_batch,
)


@cache
def install_dflash_resolution_hook():
    from sglang.srt.arg_groups import speculative_hook
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook

    # Advertise simulation capabilities only to the DFlash argument resolver.
    # The scheduler and the platform's device operations still run on CPU.
    cpu_dflash = ContextVar("cpu_dflash_simulation", default=False)
    platform = speculative_hook.current_platform
    capabilities = {
        "is_out_of_tree": lambda: True,
        "supports_speculative_algorithm": lambda algorithm: algorithm == "DFLASH",
        "supports_speculative_draft_attention_backend": lambda algorithm, backend: (
            algorithm == "DFLASH" and backend == "torch_native"
        ),
        "get_default_speculative_draft_attention_backend": lambda algorithm: (
            "torch_native"
        ),
    }

    class ResolutionPlatform:
        def __getattr__(self, name):
            if cpu_dflash.get() and name in capabilities:
                return capabilities[name]
            return getattr(platform, name)

    speculative_hook.current_platform = ResolutionPlatform()

    @register_resolution_hook("handle_speculative_decoding")
    def resolve(server_args, previous):
        cfg = resolving_view(server_args)
        simulated = cfg.speculative_algorithm == "DFLASH" and cfg.device == "cpu"
        token = cpu_dflash.set(simulated)
        try:
            previous(server_args)
        finally:
            cpu_dflash.reset(token)


class C_DFlashWorkerHook(BaseHook):
    REQUIRED = False
    HOOK_CLASS_NAME = "DFlashWorkerV2"
    HOOK_MODULE_NAME = "sglang.srt.speculative.dflash_worker_v2"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_forward = target.forward_batch_generation
        original_merge = target._maybe_merge_trained_mask_embedding

        @wraps(original_init)
        def init(self, server_args, gpu_id, nccl_port, target_worker):
            if torch.device(target_worker.device).type == "cpu":
                from sglang.srt.runtime_context import get_spec

                if (
                    not server_args.disable_overlap_schedule
                    or get_spec().speculative_draft_window_size is not None
                    or get_spec().speculative_dflash_draft_kv_ring
                ):
                    raise ValueError(
                        "DFLASH simulation requires synchronous fixed-block verification "
                        "with full draft KV"
                    )
            original_init(self, server_args, gpu_id, nccl_port, target_worker)
            if torch.device(self.device).type == "cpu":
                self._simulated_acceptance = acceptance_config(
                    "dflash", self.speculative_num_draft_tokens - 1
                )
                install_cpu_kv_slot_fallback()

        @wraps(original_merge)
        def merge_mask_embedding(self):
            if torch.device(self.device).type != "cpu":
                return original_merge(self)

        @wraps(original_forward)
        def forward(self, batch, *args, **kwargs):
            if torch.device(self.device).type != "cpu":
                return original_forward(self, batch, *args, **kwargs)

            from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

            result, bonus_tokens = simulate_speculative_batch(self, batch)
            result.next_draft_input = make_draft_input_v2(
                bonus_tokens=bonus_tokens, new_seq_lens=result.new_seq_lens
            )
            return result

        target.__init__ = init
        target._maybe_merge_trained_mask_embedding = merge_mask_embedding
        target.forward_batch_generation = forward
