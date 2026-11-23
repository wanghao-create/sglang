import asyncio
import atexit
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig

ASSETS = Path(__file__).parent / "assets"
SGLANG_ROOT = Path(__file__).parents[3]
if str(SGLANG_ROOT) not in sys.path:
    sys.path.insert(0, str(SGLANG_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def make_fixed_dataset(
    start_token: int,
    count: int,
    *,
    input_length: int = 1025,
    output_length: int = 1,
) -> SimpleDataset:
    return SimpleDataset(
        reqs=[
            GenericRequest(
                token_ids=[start_token + i] * input_length,
                input_length=input_length,
                output_length=output_length,
                custom_params={"created_time": i / 10},
            )
            for i in range(count)
        ]
    )


def _write_sim_config(tmp_path: Path) -> Path:
    table_path = tmp_path / "replay.json"
    table_path.write_text(
        json.dumps({"[[1, 1024]]": 0.001, "[[1025, 0]]": 0.01}),
        encoding="utf-8",
    )
    config = {
        "platform": {
            "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80},
            "disk_read_bandwidth_gb": 8,
            "disk_write_bandwidth_gb": 8,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 8,
        },
        "predictor": {
            "name": "replay",
            "database_path": str(table_path),
            "miss_strategy": "knn",
            "miss_knn_k": 1,
        },
        "scheduler": {"tp_size": 1, "ep_size": 1, "dp_size": 1},
    }
    config_path = tmp_path / "sim_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def make_sglang_runner(tmp_path: Path):
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(_write_sim_config(tmp_path))

    from benchmark.simulator.bench_runner import SGLangBenchmarkRunner
    from sglang.srt.server_args import ServerArgs

    runner = SGLangBenchmarkRunner(
        server_args=ServerArgs(
            model_path=str(ASSETS / "qwen3-8b"),
            load_format="dummy",
            device="cpu",
            enable_hierarchical_cache=True,
            hicache_ratio=2,
            hicache_storage_backend="file",
            hicache_storage_prefetch_policy="wait_complete",
            max_total_tokens=4 * 1024,
            page_size=256,
            skip_tokenizer_init=True,
        )
    )
    runner.clear_hicache_storage()
    return runner


def test_benchmark_sglang_runs_paged_decode(tmp_path):
    runner = make_sglang_runner(tmp_path)
    dataset = make_fixed_dataset(
        1000,
        2,
        input_length=1024,
        output_length=2,
    )

    try:
        metrics = runner.benchmark(
            BenchmarkConfig(request_rate=10, ignore_request_timestamp=False),
            dataset=dataset,
        )
        request_stats = runner.get_request_stats()
    finally:
        with patch.object(atexit, "unregister", wraps=atexit.unregister) as unregister:
            runner.shutdown()
            runner.shutdown()

        unregister.assert_called_once_with(runner.engine.shutdown)

    assert metrics["completed"] == len(dataset)
    assert metrics["total_input"] == 2 * 1024
    assert metrics["total_output"] == 2 * 2
    assert metrics["mean_tpot_ms"] > 0
    assert all(
        idx == 0 or req["created_time"] > 0 for idx, req in enumerate(request_stats)
    )


@pytest.mark.parametrize("algorithm", ["STANDALONE", "DFLASH"])
def test_speculative_commits_and_completion(monkeypatch, tmp_path, algorithm):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_MODE", "OFFLINE")
    monkeypatch.setenv("SGLANG_SIMULATOR_OUTPUT_DIR", str(tmp_path / "output"))
    config_path = tmp_path / "sim.json"
    table_path = tmp_path / "replay.json"
    step_shapes = [
        [[12, 0]] * 3,
        [[1, 12]] * 3,
        [[1, 13], [1, 14], [1, 16]],
        [[1, 14], [1, 16]],
        [[1, 15], [1, 18]],
        [[1, 16]],
        [[1, 17]],
        [[1, 18]],
    ]
    table_path.write_text(
        json.dumps({json.dumps(shape): 0.02 for shape in step_shapes})
    )
    config_path.write_text(
        json.dumps(
            {
                "platform": {
                    "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80},
                    "disk_read_bandwidth_gb": 8,
                    "disk_write_bandwidth_gb": 8,
                    "memory_read_bandwidth_gb": 64,
                    "memory_write_bandwidth_gb": 64,
                    "num_device_per_node": 1,
                },
                "predictor": {
                    "name": "replay",
                    "database_path": str(table_path),
                },
                algorithm.lower(): {
                    "accepted_draft_tokens": 0,
                    "acceptance_by_request": {"partial": [1] * 4, "full": [3, 3]},
                },
            }
        )
    )
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))

    from sglang_simulator.compat import apply_simulator_server_args
    from sglang_simulator.simulation.sglang.engine import SGLangSimulationEngine

    args = dict(
        model_path=str(ASSETS / "qwen3-8b"),
        speculative_draft_model_path=str(ASSETS / "qwen3-8b"),
        speculative_algorithm=algorithm,
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        device="cpu",
        load_format="dummy",
        skip_tokenizer_init=True,
        disable_radix_cache=True,
        max_total_tokens=256,
        context_length=128,
        page_size=1,
    )
    if algorithm == "DFLASH":
        draft = tmp_path / "draft"
        draft.mkdir()
        config = json.loads((ASSETS / "qwen3-8b" / "config.json").read_text())
        config.update(
            architectures=["DFlashDraftModel"],
            num_hidden_layers=1,
            block_size=4,
            dflash_config={"mask_token_id": 10, "target_layer_ids": [1]},
        )
        (draft / "config.json").write_text(json.dumps(config))
        args["speculative_draft_model_path"] = str(draft)
        args.pop("speculative_num_steps")
        args.pop("speculative_eagle_topk")
    apply_simulator_server_args(args)
    engine = SGLangSimulationEngine(**args)
    expected_verify_counts = {"zero": 7, "partial": 4, "full": 2}

    async def run():
        await engine.tokenizer_manager.start_profile()
        results = await asyncio.gather(
            *(
                engine.async_generate(
                    rid=rid,
                    input_ids=[11 + index] * 12,
                    sampling_params={
                        "temperature": 0,
                        "max_new_tokens": 8,
                        "ignore_eos": True,
                        "custom_params": {
                            "simulation": {"total_request": 3, "created_time": 0}
                        },
                    },
                )
                for index, rid in enumerate(expected_verify_counts)
            )
        )
        await engine.tokenizer_manager.stop_profile()
        return results

    try:
        responses = engine.loop.run_until_complete(run())
    finally:
        engine.shutdown()

    assert all(len(response["output_ids"]) == 8 for response in responses)
    assert {
        response["meta_info"]["id"]: response["meta_info"]["spec_verify_ct"]
        for response in responses
    } == expected_verify_counts
    output = tmp_path / "output"
    requests = {
        row["rid"]: row
        for row in map(json.loads, (output / "request.jsonl").read_text().splitlines())
    }
    assert set(requests) == set(expected_verify_counts)
    for rid, verify_rounds in expected_verify_counts.items():
        stats = requests[rid]
        assert stats["output_length"] == 8
        assert len(stats["gen_token_latencies"]) == 8
        assert (
            sum(latency > 0 for latency in stats["gen_token_latencies"])
            == verify_rounds + 1
        )
    iterations = list(
        map(json.loads, (output / "iteration.jsonl").read_text().splitlines())
    )
    assert [sorted(row["requests"]) for row in iterations] == step_shapes
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["replay_exact_match_steps"] == len(step_shapes)
    assert metrics["replay_miss_steps"] == 0
