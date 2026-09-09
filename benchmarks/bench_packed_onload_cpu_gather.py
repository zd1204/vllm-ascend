#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Benchmark: discrete memfabric sparse_copy vs packed CPU-gather onload.

Baseline (VLLM_ASCEND_ENABLE_CPU_GATHER_H2D=0):
    discrete host blocks -> memfabric offload.sparse_copy (many small H2D)
New path (VLLM_ASCEND_ENABLE_CPU_GATHER_H2D=1):
    packed_host_gather (OpenMP, pinned) -> one contiguous H2D copy_
    -> D2D index_copy_ scatter into topk slots.

Usage on an NPU machine:
    python benchmarks/bench_packed_onload_cpu_gather.py
    python benchmarks/bench_packed_onload_cpu_gather.py --also-sparse-copy \
        --num-entries 4096 --entry-k-bytes 1152 --entry-v-bytes 128
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch_npu


def _sync() -> None:
    torch.npu.synchronize()


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(len(s) * q) - 1))]


def _load_sparse_kv_offload_cpp():
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    npu_include_path = os.path.join(ascend_home, "include")
    npu_lib_path = os.path.join(ascend_home, "lib64")
    if not os.path.exists(npu_lib_path):
        npu_lib_path = os.path.join(ascend_home, "lib")
    torch_npu_path = os.path.dirname(torch_npu.__file__)
    src_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload.cpp",
        )
    )
    os.environ["CXX"] = os.environ.get("CXX", "clang++")
    os.environ["CC"] = os.environ.get("CC", "clang")
    return torch.utils.cpp_extension.load(
        name="sparse_kv_offload_packed_bench",
        sources=[src_path],
        extra_cflags=[
            "-O3",
            "-std=c++20",
            "-fopenmp",
            "-fPIC",
            f"-I{npu_include_path}",
            f"-I{os.path.join(torch_npu_path, 'include')}",
        ],
        extra_ldflags=[
            "-fopenmp",
            f"-L{npu_lib_path}",
            "-lascendcl",
            f"-L{os.path.join(torch_npu_path, 'lib')}",
            "-ltorch_npu",
        ],
        verbose=False,
    )


def bench(args) -> None:
    device = torch.device("npu:0")
    torch.npu.set_device(0)
    cpp = _load_sparse_kv_offload_cpp()

    num_k = args.num_entries // 2
    num_v = args.num_entries - num_k
    token_k = args.entry_k_bytes
    token_v = args.entry_v_bytes
    k_dim = token_k // 2
    v_dim = token_v // 2
    total_bytes = num_k * token_k + num_v * token_v
    num_slots_k = num_k * 2
    num_slots_v = num_v * 2

    # Discrete pinned host pool (stride-2 so source addresses are non-adjacent).
    host_pool_k = torch.empty(num_slots_k * 2, token_k, dtype=torch.uint8, pin_memory=True)
    host_pool_v = torch.empty(num_slots_v * 2, token_v, dtype=torch.uint8, pin_memory=True)
    host_pool_k.fill_(0x5A)
    host_pool_v.fill_(0x3C)

    topk_k = torch.zeros(num_slots_k, k_dim, dtype=torch.bfloat16, device=device)
    topk_v = torch.zeros(num_slots_v, v_dim, dtype=torch.bfloat16, device=device)

    srcs, dsts, sizes = [], [], []
    slots_k_list, slots_v_list = [], []
    for i in range(num_k):
        srcs.append(int(host_pool_k[i * 2].data_ptr()))
        dst_slot = (i * 2 + 1) % num_slots_k
        slots_k_list.append(dst_slot)
        dsts.append(int(topk_k.data_ptr()) + dst_slot * token_k)
        sizes.append(token_k)
    for i in range(num_v):
        srcs.append(int(host_pool_v[i * 2].data_ptr()))
        dst_slot = (i * 2 + 1) % num_slots_v
        slots_v_list.append(dst_slot)
        dsts.append(int(topk_v.data_ptr()) + dst_slot * token_v)
        sizes.append(token_v)

    src_cpu = torch.tensor(srcs, dtype=torch.int64)
    dst_cpu = torch.tensor(dsts, dtype=torch.int64)
    sizes_cpu = torch.tensor(sizes, dtype=torch.int32)
    num_entries = args.num_entries

    # ---- New path state: pinned packed buffer + NPU staging + slot metadata ----
    packed_host = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    staging_npu = torch.empty(total_bytes, dtype=torch.uint8, device=device)
    slots_k_cpu = torch.tensor(slots_k_list, dtype=torch.int64).pin_memory()
    slots_v_cpu = torch.tensor(slots_v_list, dtype=torch.int64).pin_memory()
    slots_k_npu = slots_k_cpu.to(device)
    slots_v_npu = slots_v_cpu.to(device)

    def run_packed() -> tuple[float, float, float]:
        t0 = time.perf_counter()
        packed = int(
            cpp.packed_host_gather(
                src_cpu, dst_cpu, sizes_cpu, num_entries, int(packed_host.data_ptr()), total_bytes, args.threads
            )
        )
        assert packed == total_bytes
        t1 = time.perf_counter()
        staging_npu[:total_bytes].copy_(packed_host, non_blocking=True)
        staging_k = staging_npu[: num_k * token_k].view(torch.bfloat16).view(num_k, k_dim)
        topk_k.view(-1, k_dim).index_copy_(0, slots_k_npu, staging_k)
        v_start = num_k * token_k
        staging_v = staging_npu[v_start : v_start + num_v * token_v].view(torch.bfloat16).view(num_v, v_dim)
        topk_v.view(-1, v_dim).index_copy_(0, slots_v_npu, staging_v)
        _sync()
        t2 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3

    pack_ms: list[float] = []
    h2d_scatter_ms: list[float] = []
    packed_total_ms: list[float] = []
    for i in range(args.warmup + args.iters):
        pack, h2d_scatter, total = run_packed()
        if i >= args.warmup:
            pack_ms.append(pack)
            h2d_scatter_ms.append(h2d_scatter)
            packed_total_ms.append(total)

    # Correctness: spot-check one K and one V slot.
    expect_k = host_pool_k[0].view(torch.bfloat16)
    assert torch.equal(topk_k[slots_k_list[0]].cpu(), expect_k), "packed path K mismatch"
    expect_v = host_pool_v[0].view(torch.bfloat16)
    assert torch.equal(topk_v[slots_v_list[0]].cpu(), expect_v), "packed path V mismatch"

    print(f"entries={num_entries} (k={num_k}x{token_k}B, v={num_v}x{token_v}B), "
          f"packed_bytes={total_bytes / 1024:.1f} KiB, threads={args.threads}")
    print(f"[packed] cpu_gather: mean={statistics.mean(pack_ms):.3f} ms "
          f"p50={_percentile(pack_ms, 0.5):.3f} p99={_percentile(pack_ms, 0.99):.3f}")
    print(f"[packed] h2d+scatter: mean={statistics.mean(h2d_scatter_ms):.3f} ms "
          f"p50={_percentile(h2d_scatter_ms, 0.5):.3f} p99={_percentile(h2d_scatter_ms, 0.99):.3f}")
    print(f"[packed] total: mean={statistics.mean(packed_total_ms):.3f} ms "
          f"p50={_percentile(packed_total_ms, 0.5):.3f} p99={_percentile(packed_total_ms, 0.99):.3f}")
    gib = total_bytes / (1 << 30)
    print(f"[packed] effective bandwidth: {gib / (statistics.mean(packed_total_ms) / 1e3):.2f} GiB/s")

    if args.also_sparse_copy:
        from memfabric_hybrid import offload

        cfg = offload.OffloadConfig()
        cfg.device_id = 0
        cfg.reserve_size = 64 << 20
        cfg.alloc_size = 64 << 20
        cfg.world_size = 1
        cfg.rank_id = 0
        cfg.scene = offload.Scene.LOCAL
        assert offload.initialize(cfg) == 0
        try:
            src_npu = src_cpu.to(device)
            dst_npu = dst_cpu.to(device)
            sizes_npu = sizes_cpu.to(device)
            num_npu = torch.tensor([num_entries], dtype=torch.int32, device=device)
            sparse_ms: list[float] = []
            for i in range(args.warmup + args.iters):
                _sync()
                t0 = time.perf_counter()
                rc = offload.sparse_copy(src_npu, dst_npu, sizes_npu, num_npu, device)
                _sync()
                assert rc in (None, 0)
                if i >= args.warmup:
                    sparse_ms.append((time.perf_counter() - t0) * 1e3)
            print(f"[sparse_copy] total: mean={statistics.mean(sparse_ms):.3f} ms "
                  f"p50={_percentile(sparse_ms, 0.5):.3f} p99={_percentile(sparse_ms, 0.99):.3f}")
            print(f"[sparse_copy] effective bandwidth: {gib / (statistics.mean(sparse_ms) / 1e3):.2f} GiB/s")
            speedup = statistics.mean(sparse_ms) / statistics.mean(packed_total_ms)
            print(f"[compare] packed path is {speedup:.2f}x the discrete sparse_copy path")
        finally:
            offload.uninitialize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-entries", type=int, default=2048, help="total k+v entries per step")
    parser.add_argument("--entry-k-bytes", type=int, default=1152, help="bytes per K token (e.g. 576*2)")
    parser.add_argument("--entry-v-bytes", type=int, default=128, help="bytes per V token (e.g. 64*2)")
    parser.add_argument("--threads", type=int, default=4, help="OpenMP threads for CPU gather")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--also-sparse-copy", action="store_true", help="also bench memfabric sparse_copy baseline")
    args = parser.parse_args()
    bench(args)


if __name__ == "__main__":
    main()
