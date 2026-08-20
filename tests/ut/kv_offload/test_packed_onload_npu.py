# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU correctness check: TP0 GVA pack + sparse_copy n=1 H2D + D2D scatter."""

from __future__ import annotations

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
offload = pytest.importorskip("memfabric_hybrid").offload

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or torch.npu.device_count() <= 0,
    reason="NPU required for packed onload verification",
)


def _load_sparse_kv_offload_cpp():
    import torch_npu as torch_npu_mod

    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    npu_include_path = os.path.join(ascend_home, "include")
    npu_lib_path = os.path.join(ascend_home, "lib64")
    if not os.path.exists(npu_lib_path):
        npu_lib_path = os.path.join(ascend_home, "lib")
    torch_npu_path = os.path.dirname(torch_npu_mod.__file__)
    src_path = os.path.join(
        os.path.dirname(__file__),
        "../../../vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload.cpp",
    )
    src_path = os.path.abspath(src_path)
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


def test_packed_host_gather_contiguous_h2d_and_d2d():
    device = torch.device("npu:0")
    torch.npu.set_device(0)

    num_entries = 8
    entry_bytes = 256
    total_bytes = num_entries * entry_bytes

    cfg = offload.OffloadConfig()
    cfg.device_id = 0
    cfg.reserve_size = 64 << 20
    cfg.alloc_size = 64 << 20
    cfg.world_size = 1
    cfg.rank_id = 0
    cfg.scene = offload.Scene.LOCAL
    assert offload.initialize(cfg) == 0
    cpp = None
    try:
        cpp = _load_sparse_kv_offload_cpp()
        host_pool = torch.empty(num_entries * 2, entry_bytes, dtype=torch.uint8, pin_memory=True)
        for i in range(num_entries):
            host_pool[i * 2].fill_(i + 1)

        packed_host = offload.empty([total_bytes], dtype=torch.uint8, pin_memory=True)
        device_pool = torch.zeros(num_entries * 2, entry_bytes, dtype=torch.uint8, device=device)
        device_staging = torch.zeros(total_bytes, dtype=torch.uint8, device=device)

        src = torch.tensor([int(host_pool[i * 2].data_ptr()) for i in range(num_entries)], dtype=torch.int64)
        dst = torch.tensor([int(device_pool[i * 2].data_ptr()) for i in range(num_entries)], dtype=torch.int64)
        sizes = torch.full((num_entries,), entry_bytes, dtype=torch.int32)

        packed_bytes = int(
            cpp.packed_host_gather(
                src,
                dst,
                sizes,
                num_entries,
                int(packed_host.data_ptr()),
                total_bytes,
                4,
            )
        )
        assert packed_bytes == total_bytes
        packed_cpu = packed_host.cpu() if packed_host.device.type != "cpu" else packed_host
        for i in range(num_entries):
            expect = host_pool[i * 2].cpu()
            got = packed_cpu.view(-1)[i * entry_bytes : (i + 1) * entry_bytes]
            assert torch.equal(got, expect), f"host pack mismatch at entry {i}"

        assert cpp.packed_contiguous_h2d(
            int(packed_host.data_ptr()),
            int(device_staging.data_ptr()),
            total_bytes,
        )
        assert cpp.packed_d2d_scatter(
            src,
            dst,
            sizes,
            num_entries,
            int(device_staging.data_ptr()),
            total_bytes,
        )
        torch.npu.synchronize()

        for i in range(num_entries):
            assert torch.equal(device_pool[i * 2].cpu(), host_pool[i * 2].cpu()), f"D2D mismatch at entry {i}"
    finally:
        offload.uninitialize()
