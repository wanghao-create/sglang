"""Simulate token acceptance for the native synchronous STANDALONE worker."""

from functools import wraps

import torch
from sglang_simulator.hook import BaseHook
from sglang_simulator.simulation.sglang.speculative import (
    acceptance_config,
    install_cpu_kv_slot_fallback,
    simulate_speculative_batch,
)


class C_StandaloneWorkerHook(BaseHook):
    REQUIRED = False
    HOOK_CLASS_NAME = "StandaloneWorkerV2"
    HOOK_MODULE_NAME = "sglang.srt.speculative.standalone_worker_v2"

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_forward = target.forward_batch_generation
        original_init_attention = target.init_attention_backends

        @wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if torch.device(self.device).type != "cpu":
                return
            if (
                self.topk != 1
                or self.speculative_num_draft_tokens != self.speculative_num_steps + 1
                or not self.server_args.disable_overlap_schedule
                or self.server_args.speculative_adaptive
            ):
                raise ValueError(
                    "STANDALONE simulation requires a fixed top-k=1 chain "
                    "with speculative_num_draft_tokens=speculative_num_steps+1 "
                    "and overlap disabled"
                )
            self._simulated_acceptance = acceptance_config(
                "standalone", self.speculative_num_steps
            )
            install_cpu_kv_slot_fallback()

        @wraps(original_init_attention)
        def init_attention_backends(self):
            if torch.device(self.device).type != "cpu":
                return original_init_attention(self)

            # As with the target runner, simulated drafts need no attention kernel.
            self.draft_worker.draft_worker.init_attention_backends()
            self.draft_worker.draft_attn_backend = None
            self.draft_worker.draft_extend_attn_backend = None

        @wraps(original_forward)
        def forward(self, batch, *args, **kwargs):
            if torch.device(self.device).type != "cpu":
                return original_forward(self, batch, *args, **kwargs)

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

        target.__init__ = init
        target.init_attention_backends = init_attention_backends
        target.forward_batch_generation = forward
