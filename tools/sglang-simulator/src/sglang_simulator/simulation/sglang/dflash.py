"""Simulate accepted drafts in the native synchronous DFlash worker."""

from __future__ import annotations

from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.sglang.speculative import (
    acceptance_config,
    simulate_speculative_batch,
)

_RESOLUTION_HOOK_INSTALLED = False


class _SimDFlashPlatform:
    """Advertise simulated DFlash support only during ServerArgs resolution."""

    def __init__(self, platform):
        self._platform = platform

    def is_out_of_tree(self) -> bool:
        return True

    def supports_speculative_algorithm(self, algorithm: str) -> bool:
        return algorithm == "DFLASH"

    def supports_speculative_draft_attention_backend(
        self, algorithm: str, backend: str
    ) -> bool:
        return algorithm == "DFLASH" and backend == "torch_native"

    def get_default_speculative_draft_attention_backend(self, algorithm: str) -> str:
        return "torch_native"

    def __getattr__(self, name: str):
        return getattr(self._platform, name)


def install_dflash_resolution_hook() -> None:
    """Register the simulated DFlash ServerArgs resolution hook once."""
    global _RESOLUTION_HOOK_INSTALLED
    if _RESOLUTION_HOOK_INSTALLED:
        return

    try:
        from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook
    except ImportError:
        return

    @register_resolution_hook("handle_speculative_decoding")
    def wrapped_handle_speculative_decoding(server_args, previous):
        from sglang.srt.arg_groups.model_override_base import resolving_view

        cfg = resolving_view(server_args)
        if (cfg.speculative_algorithm or "").upper() != "DFLASH":
            return previous(server_args)

        from sglang.srt.arg_groups import speculative_hook

        original_platform = speculative_hook.current_platform
        speculative_hook.current_platform = _SimDFlashPlatform(original_platform)
        try:
            return previous(server_args)
        finally:
            speculative_hook.current_platform = original_platform

    _RESOLUTION_HOOK_INSTALLED = True


class C_DFlashWorkerHook(BaseHook):
    HOOK_CLASS_NAME = "DFlashWorkerV2"
    HOOK_MODULE_NAME = "sglang.srt.speculative.dflash_worker_v2"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, server_args, gpu_id, nccl_port, target_worker):
            from sglang.srt.runtime_context import get_spec

            if (
                get_spec().speculative_draft_window_size is not None
                or get_spec().speculative_dflash_draft_kv_ring
            ):
                raise ValueError(
                    "DFLASH simulation requires fixed-block verification with full draft KV."
                )
            original_init(self, server_args, gpu_id, nccl_port, target_worker)
            self._simulated_acceptance = acceptance_config(
                "dflash", self.speculative_num_draft_tokens - 1
            )

        def override_maybe_merge_trained_mask_embedding(self):
            return None

        def override_forward_batch_generation(self, batch, *args, **kwargs):
            from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

            result, bonus_tokens = simulate_speculative_batch(self, batch)
            result.next_draft_input = make_draft_input_v2(
                bonus_tokens=bonus_tokens, new_seq_lens=result.new_seq_lens
            )
            return result

        target.__init__ = wrapped_init
        target._maybe_merge_trained_mask_embedding = (
            override_maybe_merge_trained_mask_embedding
        )
        target.forward_batch_generation = override_forward_batch_generation
