import sys
import types


def _assign_cpu_kv_slots(
    req_indices, req_to_token, starts, ends, slots, _width
) -> None:
    offset = 0
    for index, start, end in zip(
        req_indices.tolist(), starts.tolist(), ends.tolist()
    ):
        length = end - start
        req_to_token[index, start:end] = slots[offset : offset + length]
        offset += length


def install_cpu_kv_slot_fallback() -> None:
    """Provide a Python fallback when sgl_kernel lacks the CPU slot-assign op."""
    import torch

    if hasattr(getattr(torch.ops, "sgl_kernel", None), "assign_req_to_token_pool_cpu"):
        return

    try:
        import sgl_kernel
    except ImportError:
        sgl_kernel = None

    if sgl_kernel is not None:
        sgl_kernel.assign_req_to_token_pool_cpu = _assign_cpu_kv_slots

    allocation = sys.modules.get("sglang.srt.mem_cache.allocation")
    if allocation is not None:
        allocation.assign_req_to_token_pool_cpu = _assign_cpu_kv_slots


def install_load_utils_stub() -> None:
    """Install the kernel loader stub before importing the sgl_kernel package."""
    module_name = "sgl_kernel.load_utils"
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__package__ = "sgl_kernel"
        sys.modules[module_name] = module

    module._load_architecture_specific_ops = lambda *args, **kwargs: None
    module._preload_cuda_library = lambda *args, **kwargs: None
