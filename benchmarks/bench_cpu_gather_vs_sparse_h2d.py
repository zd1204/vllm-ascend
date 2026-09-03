#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compare Sparse KV H2D before/after CPU gather aggregation.

聚合前: swap_blocks_batch H2D (discrete host blocks -> discrete device blocks)
聚合后: CpuGatherPool + single-entry swap_blocks_batch(H2D) + swap_blocks_batch(D2D)

Also optionally measures memfabric offload.sparse_copy when --also-sparse-copy is set.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch_npu

from vllm_ascend.kv_offload.cpu_gather import CpuGatherPool, GatherItem
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op


def _sync() -> None:
    torch.npu.synchronize()


def _p50_p99(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2], s[max(0, int(n * 0.99) - 1)]


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def _contiguous_h2d(host_base: int, device_base: int, nbytes: int) -> None:
    """One large H2D on the current NPU stream."""
    if nbytes <= 0:
        return
    # Keep benchmark on the same verified path as the minimal single-op test:
    # one contiguous H2D expressed as a single swap_blocks_batch entry.
    torch.ops._C_ascend.swap_blocks_batch(
        torch.tensor([host_base], dtype=torch.int64),
        torch.tensor([device_base], dtype=torch.int64),
        torch.tensor([nbytes], dtype=torch.int64),
        0,
    )


def bench_case(
    *,
    num_entries: int,
    entry_bytes: int,
    gather_threads: int,
    warmup: int,
    iters: int,
    device: torch.device,
    also_sparse_copy: bool,
) -> dict:
    total_bytes = num_entries * entry_bytes

    # Discrete host sources (pinned), stride-2 so addresses are non-adjacent.
    host_pool = torch.empty(num_entries * 2, entry_bytes, dtype=torch.uint8, pin_memory=True)
    for i in range(num_entries):
        host_pool[i * 2].fill_(i & 0xFF)

    device_pool = torch.empty(num_entries * 2, entry_bytes, dtype=torch.uint8, device=device)
    device_pool.zero_()

    src_list = [int(host_pool[i * 2].data_ptr()) for i in range(num_entries)]
    dst_list = [int(device_pool[i * 2].data_ptr()) for i in range(num_entries)]
    src_cpu = torch.tensor(src_list, dtype=torch.int64)
    dst_cpu = torch.tensor(dst_list, dtype=torch.int64)
    sizes_cpu = torch.full((num_entries,), entry_bytes, dtype=torch.int64)

    # ---- 聚合前: discrete H2D ----
    discrete_ms: list[float] = []
    for i in range(warmup + iters):
        device_pool.zero_()
        _sync()
        t0 = time.perf_counter()
        torch.ops._C_ascend.swap_blocks_batch(src_cpu, dst_cpu, sizes_cpu, 0)
        _sync()
        if i >= warmup:
            discrete_ms.append((time.perf_counter() - t0) * 1e3)

    if not torch.equal(device_pool[0].cpu(), host_pool[0].cpu()):
        raise RuntimeError("discrete H2D correctness check failed")

    # ---- 聚合后: gather + contiguous H2D + D2D scatter ----
    host_gather = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    device_staging = torch.empty(total_bytes, dtype=torch.uint8, device=device)
    pool = CpuGatherPool(num_threads=gather_threads)
    host_base = int(host_gather.data_ptr())
    device_base = int(device_staging.data_ptr())
    items = [
        GatherItem(src_ptr=src_list[i], dst_offset=i * entry_bytes, size=entry_bytes)
        for i in range(num_entries)
    ]
    staging_srcs = torch.tensor(
        [device_base + i * entry_bytes for i in range(num_entries)],
        dtype=torch.int64,
    )
    staging_dsts = dst_cpu.clone()
    staging_sizes = sizes_cpu.clone()

    gather_ms: list[float] = []
    try:
        for i in range(warmup + iters):
            device_pool.zero_()
            _sync()
            t0 = time.perf_counter()
            pool.gather(items, host_base)
            _contiguous_h2d(host_base, device_base, total_bytes)
            torch.ops._C_ascend.swap_blocks_batch(staging_srcs, staging_dsts, staging_sizes, 2)
            _sync()
            if i >= warmup:
                gather_ms.append((time.perf_counter() - t0) * 1e3)
    finally:
        pool.close()

    last = num_entries - 1
    if not torch.equal(device_pool[last * 2].cpu(), host_pool[last * 2].cpu()):
        raise RuntimeError("gather path correctness check failed")

    sparse_ms: list[float] = []
    if also_sparse_copy:
        from memfabric_hybrid import offload

        src_npu = src_cpu.to(device)
        dst_npu = dst_cpu.to(device)
        sizes_npu = sizes_cpu.to(dtype=torch.int32, device=device)
        num_npu = torch.tensor(num_entries, dtype=torch.int32, device=device)
        for i in range(warmup + iters):
            device_pool.zero_()
            _sync()
            t0 = time.perf_counter()
            rc = offload.sparse_copy(src_npu, dst_npu, sizes_npu, num_npu, device)
            _sync()
            if rc not in (None, 0):
                raise RuntimeError(f"sparse_copy failed rc={rc}")
            if i >= warmup:
                sparse_ms.append((time.perf_counter() - t0) * 1e3)

    d_mean = _mean(discrete_ms)
    g_mean = _mean(gather_ms)
    dp50, dp99 = _p50_p99(discrete_ms)
    gp50, gp99 = _p50_p99(gather_ms)
    row = {
        "num_entries": num_entries,
        "entry_bytes": entry_bytes,
        "total_MB": total_bytes / (1024 * 1024),
        "discrete_mean_ms": d_mean,
        "discrete_p50_ms": dp50,
        "discrete_p99_ms": dp99,
        "gather_mean_ms": g_mean,
        "gather_p50_ms": gp50,
        "gather_p99_ms": gp99,
        "speedup_mean": d_mean / g_mean if g_mean > 0 else float("inf"),
        "discrete_GBps": (total_bytes / 1e9) / (d_mean / 1e3) if d_mean > 0 else 0.0,
        "gather_GBps": (total_bytes / 1e9) / (g_mean / 1e3) if g_mean > 0 else 0.0,
    }
    if sparse_ms:
        s_mean = _mean(sparse_ms)
        sp50, sp99 = _p50_p99(sparse_ms)
        row.update(
            {
                "sparse_mean_ms": s_mean,
                "sparse_p50_ms": sp50,
                "sparse_p99_ms": sp99,
                "sparse_GBps": (total_bytes / 1e9) / (s_mean / 1e3) if s_mean > 0 else 0.0,
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--entry-bytes", type=int, default=1024)
    parser.add_argument("--also-sparse-copy", action="store_true")
    parser.add_argument("--pool-gb", type=float, default=2.0)
    parser.add_argument(
        "--entries",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096],
    )
    args = parser.parse_args()

    bootstrap_custom_op_env(include_vendor_lib=True)
    assert enable_custom_op(), "custom ops required"

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    if args.also_sparse_copy:
        from memfabric_hybrid import offload

        cfg = offload.OffloadConfig()
        cfg.device_id = args.device
        cfg.reserve_size = int(args.pool_gb * (1 << 30))
        cfg.alloc_size = int(args.pool_gb * (1 << 30))
        cfg.world_size = 1
        cfg.rank_id = 0
        cfg.scene = offload.Scene.LOCAL
        assert offload.initialize(cfg) == 0

    print(
        f"device=npu:{args.device} threads={args.threads} "
        f"entry_bytes={args.entry_bytes} warmup={args.warmup} iters={args.iters}"
    )
    print(
        f"{'entries':>8} {'MB':>8} {'discrete_ms':>12} {'gather_ms':>10} "
        f"{'speedup':>8} {'disc_GBps':>10} {'gath_GBps':>10}"
    )

    rows = []
    try:
        for n in args.entries:
            row = bench_case(
                num_entries=n,
                entry_bytes=args.entry_bytes,
                gather_threads=args.threads,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
                also_sparse_copy=args.also_sparse_copy,
            )
            rows.append(row)
            print(
                f"{row['num_entries']:8d} {row['total_MB']:8.2f} "
                f"{row['discrete_mean_ms']:12.3f} {row['gather_mean_ms']:10.3f} "
                f"{row['speedup_mean']:8.2f}x {row['discrete_GBps']:10.2f} {row['gather_GBps']:10.2f}",
                flush=True,
            )
    finally:
        if args.also_sparse_copy:
            from memfabric_hybrid import offload

            offload.uninitialize()

    print("\n# p50/p99 (ms)  聚合前=discrete H2D  聚合后=gather+contig H2D+D2D")
    for row in rows:
        print(
            f"entries={row['num_entries']}: "
            f"discrete p50={row['discrete_p50_ms']:.3f} p99={row['discrete_p99_ms']:.3f} | "
            f"gather p50={row['gather_p50_ms']:.3f} p99={row['gather_p99_ms']:.3f}"
        )


if __name__ == "__main__":
    main()
