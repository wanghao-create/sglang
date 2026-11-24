"""Simulate token acceptance for the native synchronous STANDALONE worker."""

from __future__ import annotations

import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.sglang.speculative import (
    acceptance_config,
    simulate_speculative_batch,
)


class C_StandaloneWorkerHook(BaseHook):
    HOOK_CLASS_NAME = "StandaloneWorkerV2"
    HOOK_MODULE_NAME = "sglang.srt.speculative.standalone_worker_v2"
    REQUIRED = False

    @classmethod
    def hook(cls, target):
        original_init = target.__init__

        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if self.topk != 1 or self.server_args.speculative_adaptive:
                raise ValueError(
                    "STANDALONE simulation requires fixed-width top-k=1 speculation."
                )
            self._simulated_acceptance = acceptance_config(
                "standalone", self.speculative_num_draft_tokens - 1
            )

        def override_init_attention_backends(self):
            # As with the target runner, simulated drafts need no attention kernel.
            self.draft_worker.draft_worker.init_attention_backends()
            self.draft_worker.draft_attn_backend = None
            self.draft_worker.draft_extend_attn_backend = None

        def override_forward_batch_generation(self, batch, *args, **kwargs):
            from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
            from sglang.srt.speculative.eagle_info import EagleDraftInput

            result, bonus_tokens = simulate_speculative_batch(self, batch)

            # The scheduler still filters/merges the native draft state, even
            # though the simulator does not use proposal scores or hidden states.
            result.next_draft_input = EagleDraftInput(
                bonus_tokens=bonus_tokens,
                topk_p=torch.ones(
                    (len(batch.reqs), 1), dtype=torch.float32, device=self.device
                ),
                topk_index=bonus_tokens.long().unsqueeze(1),
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            return result

        target.__init__ = wrapped_init
        target.init_attention_backends = override_init_attention_backends
        target.forward_batch_generation = override_forward_batch_generation
