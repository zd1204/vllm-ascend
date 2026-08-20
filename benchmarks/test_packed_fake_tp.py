#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fake 2-TP packed-onload test on one NPU.

Modes:
  same-process  One card, two H2D channels after TP0 pack:
                dest_tp0 = ACL contiguous H2D (real TP0)
                dest_tp1 = sparse_copy(n=1) from the same packed GVA (fake TP1)
  two-proc      Two processes, SHARED world_size=2, both pinned to npu:0.
                Rank0 packs into SHARED GVA; rank1 sparse_copy(n=1).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from multiprocessing import Queue, get_context

NUM_ENTRIES = 8
ENTRY_BYTES = 256
TOTAL_BYTES = NUM_ENTRIES * ENTRY_BYTES


def _load_cpp():
    import torch
    import torch.utils.cpp_extension  # noqa: F401
    import torch_npu

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
        name="sparse_kv_offload_packed_verify_v2",
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


def _make_host_blocks():
    import torch

    host_pool = torch.empty(NUM_ENTRIES * 2, ENTRY_BYTES, dtype=torch.uint8, pin_memory=True)
    for i in range(NUM_ENTRIES):
        host_pool[i * 2].fill_(i + 1)
    src = torch.tensor([int(host_pool[i * 2].data_ptr()) for i in range(NUM_ENTRIES)], dtype=torch.int64)
    sizes = torch.full((NUM_ENTRIES,), ENTRY_BYTES, dtype=torch.int32)
    return host_pool, src, sizes


def _check_dest(name, dest, host_pool):
    import torch

    for i in range(NUM_ENTRIES):
        if not torch.equal(dest[i * 2].cpu(), host_pool[i * 2].cpu()):
            got = dest[i * 2].cpu()[:8].tolist()
            exp = host_pool[i * 2].cpu()[:8].tolist()
            raise AssertionError(f"{name} mismatch entry {i}: got={got} expect={exp}")
    print(f"[ok] {name} matches host source")


def run_same_process() -> None:
    import torch
    import torch_npu
    from memfabric_hybrid import offload

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    cpp = _load_cpp()

    cfg = offload.OffloadConfig()
    cfg.device_id = 0
    cfg.reserve_size = 64 << 20
    cfg.alloc_size = 64 << 20
    cfg.world_size = 1
    cfg.rank_id = 0
    cfg.scene = offload.Scene.SHARED
    rc = offload.initialize(cfg)
    print(f"[same-process] SHARED init rc={rc} (0=ok)")
    if rc != 0:
        raise RuntimeError(f"offload.initialize failed rc={rc}")

    try:
        host_pool, src, sizes = _make_host_blocks()
        packed = offload.empty([TOTAL_BYTES], dtype=torch.uint8, pin_memory=True)
        dst_dummy = torch.ones(NUM_ENTRIES, dtype=torch.int64)  # nonzero so pack keeps entries
        packed_bytes = int(
            cpp.packed_host_gather(src, dst_dummy, sizes, NUM_ENTRIES, int(packed.data_ptr()), TOTAL_BYTES, 4)
        )
        assert packed_bytes == TOTAL_BYTES, packed_bytes
        print(f"[same-process] packed GVA=0x{int(packed.data_ptr()):x} nbytes={packed_bytes}")

        staging0 = torch.zeros(TOTAL_BYTES, dtype=torch.uint8, device=device)
        dest0 = torch.zeros(NUM_ENTRIES * 2, ENTRY_BYTES, dtype=torch.uint8, device=device)
        dst0 = torch.tensor([int(dest0[i * 2].data_ptr()) for i in range(NUM_ENTRIES)], dtype=torch.int64)
        cpp.packed_contiguous_h2d(int(packed.data_ptr()), int(staging0.data_ptr()), TOTAL_BYTES)
        cpp.packed_d2d_scatter(src, dst0, sizes, NUM_ENTRIES, int(staging0.data_ptr()), TOTAL_BYTES)
        torch.npu.synchronize()
        _check_dest("TP0 ACL H2D+D2D", dest0, host_pool)

        staging1 = torch.zeros(TOTAL_BYTES, dtype=torch.uint8, device=device)
        dest1 = torch.zeros(NUM_ENTRIES * 2, ENTRY_BYTES, dtype=torch.uint8, device=device)
        src_npu = torch.tensor([int(packed.data_ptr())], dtype=torch.int64, device=device)
        dst_npu = torch.tensor([int(staging1.data_ptr())], dtype=torch.int64, device=device)
        size_npu = torch.tensor([TOTAL_BYTES], dtype=torch.int32, device=device)
        num_npu = torch.tensor(1, dtype=torch.int32, device=device)
        sc_rc = offload.sparse_copy(src_npu, dst_npu, size_npu, num_npu, device)
        torch.npu.synchronize()
        uniq = staging1.cpu().unique().tolist()
        print(f"[same-process] fake-TP1 sparse_copy(n=1) rc={sc_rc} staging_unique={uniq[:8]}")
        if uniq == [0]:
            print("[same-process] FAKE TP1 FAIL: sparse_copy n=1 did not move data (often missing libhcom.so)")
            return
        dst1 = torch.tensor([int(dest1[i * 2].data_ptr()) for i in range(NUM_ENTRIES)], dtype=torch.int64)
        cpp.packed_d2d_scatter(src, dst1, sizes, NUM_ENTRIES, int(staging1.data_ptr()), TOTAL_BYTES)
        torch.npu.synchronize()
        _check_dest("fake-TP1 sparse_copy H2D+D2D", dest1, host_pool)
    finally:
        offload.uninitialize()


def _two_proc_worker(rank: int, world: int, result_q: Queue) -> None:
    try:
        import torch
        import torch_npu
        from memfabric_hybrid import offload

        torch.npu.set_device(0)
        device = torch.device("npu:0")
        cfg = offload.OffloadConfig()
        cfg.device_id = 0
        cfg.reserve_size = 64 << 20
        cfg.alloc_size = (64 << 20) if rank == 0 else 0
        cfg.world_size = world
        cfg.rank_id = rank
        cfg.scene = offload.Scene.SHARED
        t0 = time.time()
        rc = offload.initialize(cfg)
        result_q.put(("init", rank, rc, time.time() - t0))
        if rc != 0:
            return

        if rank == 0:
            cpp = _load_cpp()
            host_pool, src, sizes = _make_host_blocks()
            packed = offload.empty([TOTAL_BYTES], dtype=torch.uint8, pin_memory=True)
            dst_dummy = torch.ones(NUM_ENTRIES, dtype=torch.int64)
            packed_bytes = int(
                cpp.packed_host_gather(src, dst_dummy, sizes, NUM_ENTRIES, int(packed.data_ptr()), TOTAL_BYTES, 4)
            )
            gva = int(packed.data_ptr())
            result_q.put(("packed", rank, gva, packed_bytes))
            # Keep process alive so rank1 can DMA; wait for rank1 done via file.
            done = "/tmp/vllm_ascend_fake_tp_rank1_done"
            if os.path.exists(done):
                os.remove(done)
            deadline = time.time() + 60
            while time.time() < deadline and not os.path.exists(done):
                time.sleep(0.2)
            offload.uninitialize()
            result_q.put(("rank0_exit", rank, os.path.exists(done), 0))
        else:
            # Rank1 waits for packed gva from queue via a side file written by parent... 
            # Parent forwards gva through a well-known file.
            gva_file = "/tmp/vllm_ascend_fake_tp_packed_gva"
            deadline = time.time() + 60
            while time.time() < deadline and not os.path.exists(gva_file):
                time.sleep(0.2)
            if not os.path.exists(gva_file):
                result_q.put(("rank1_timeout_gva", rank, 0, 0))
                offload.uninitialize()
                return
            gva = int(open(gva_file).read().strip())
            staging = torch.zeros(TOTAL_BYTES, dtype=torch.uint8, device=device)
            src_npu = torch.tensor([gva], dtype=torch.int64, device=device)
            dst_npu = torch.tensor([int(staging.data_ptr())], dtype=torch.int64, device=device)
            size_npu = torch.tensor([TOTAL_BYTES], dtype=torch.int32, device=device)
            num_npu = torch.tensor(1, dtype=torch.int32, device=device)
            sc_rc = offload.sparse_copy(src_npu, dst_npu, size_npu, num_npu, device)
            torch.npu.synchronize()
            uniq = staging.cpu().unique().tolist()
            result_q.put(("rank1_sparse", rank, sc_rc, uniq[:8]))
            open("/tmp/vllm_ascend_fake_tp_rank1_done", "w").write("1")
            offload.uninitialize()
    except Exception:
        result_q.put(("error", rank, traceback.format_exc(), 0))


def run_two_proc(timeout_s: float) -> None:
    for path in ("/tmp/vllm_ascend_fake_tp_packed_gva", "/tmp/vllm_ascend_fake_tp_rank1_done"):
        if os.path.exists(path):
            os.remove(path)

    # spawn: parent already imported torch_npu in same-process mode; fork cannot
    # re-init NPU in the child.
    ctx = get_context("spawn")
    result_q = ctx.Queue()
    procs = [ctx.Process(target=_two_proc_worker, args=(rank, 2, result_q), daemon=True) for rank in (0, 1)]
    print("[two-proc] starting 2 processes on npu:0 SHARED world_size=2")
    for p in procs:
        p.start()

    deadline = time.time() + timeout_s
    events = []
    packed_gva = None
    while time.time() < deadline:
        try:
            ev = result_q.get(timeout=0.5)
        except Exception:
            if not any(p.is_alive() for p in procs) and result_q.empty():
                break
            continue
        print("[two-proc] event", ev[0], "rank", ev[1], ev[2:])
        events.append(ev)
        if ev[0] == "packed":
            packed_gva = ev[2]
            with open("/tmp/vllm_ascend_fake_tp_packed_gva", "w") as f:
                f.write(str(packed_gva))
        if ev[0] in ("rank1_sparse", "error", "rank1_timeout_gva"):
            break

    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)
            print(f"[two-proc] killed hung pid={p.pid} rank leftover")

    inits = [e for e in events if e[0] == "init"]
    print(f"[two-proc] init events={inits}")
    sparse = [e for e in events if e[0] == "rank1_sparse"]
    if not inits:
        print("[two-proc] FAIL: SHARED initialize did not return (likely needs HCCL/libhcom + 2 devices)")
        return
    if any(e[2] != 0 for e in inits):
        print("[two-proc] FAIL: initialize rc != 0; cannot fake 2 TP on one card via memfabric SHARED")
        return
    if sparse:
        sc_rc, uniq = sparse[0][2], sparse[0][3]
        print(f"[two-proc] rank1 sparse_copy rc={sc_rc} unique={uniq}")
        if uniq == [0] or uniq == []:
            print("[two-proc] FAIL: rank1 H2D empty")
        else:
            print("[two-proc] PASS: rank1 saw packed data via sparse_copy")
    else:
        print("[two-proc] FAIL: rank1 never completed sparse_copy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["same-process", "two-proc", "both"], default="both")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    if args.mode in ("same-process", "both"):
        print("===== same-process: 1 card, ACL vs sparse_copy(n=1) =====")
        run_same_process()
    if args.mode in ("two-proc", "both"):
        print("===== two-proc: 1 card, 2 processes, SHARED =====")
        run_two_proc(args.timeout)


if __name__ == "__main__":
    main()
