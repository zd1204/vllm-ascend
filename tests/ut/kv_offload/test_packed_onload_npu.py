# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NPU correctness check for the packed onload path without memfabric:
CPU gather into a plain pinned buffer -> one contiguous H2D copy_ into an
NPU staging buffer -> D2D index_copy_ scatter into topk slots."""

from __future__ import annotations

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

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
        name="sparse_kv_offload_packed_verify_cpu_path",
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


def _make_descriptors(num_k: int, num_v: int, token_k: int, token_v: int, entry_pattern: int):
    """Build a discrete host pool and descriptors in the packed-onload
    contract: entries [0, num_k) are K, [num_k, num_k + num_v) are V."""
    num_slots_k = 16
    num_slots_v = 8
    host_pool_k = torch.empty(num_slots_k, token_k, dtype=torch.uint8, pin_memory=True)
    host_pool_v = torch.empty(num_slots_v, token_v, dtype=torch.uint8, pin_memory=True)
    for i in range(num_slots_k):
        host_pool_k[i].fill_((i * entry_pattern + 1) & 0xFF)
    for i in range(num_slots_v):
        host_pool_v[i].fill_((i * entry_pattern + 101) & 0xFF)

    # Source slots in the host pool (discrete, shuffled on purpose).
    src_slots_k = [3, 0, 7, 11][:num_k]
    src_slots_v = [5, 1, 2][:num_v]
    # Destination slots in the device topk buffers.
    dst_slots_k = [9, 2, 14, 4][:num_k]
    dst_slots_v = [6, 0, 3][:num_v]

    topk_k = torch.zeros(num_slots_k, token_k // 2, dtype=torch.bfloat16, device="npu")
    topk_v = torch.zeros(num_slots_v, token_v // 2, dtype=torch.bfloat16, device="npu")

    srcs, dsts, sizes = [], [], []
    for src_slot, dst_slot in zip(src_slots_k, dst_slots_k):
        srcs.append(int(host_pool_k[src_slot].data_ptr()))
        dsts.append(int(topk_k.data_ptr()) + dst_slot * token_k)
        sizes.append(token_k)
    for src_slot, dst_slot in zip(src_slots_v, dst_slots_v):
        srcs.append(int(host_pool_v[src_slot].data_ptr()))
        dsts.append(int(topk_v.data_ptr()) + dst_slot * token_v)
        sizes.append(token_v)

    desc = {
        "src": torch.tensor(srcs, dtype=torch.int64),
        "dst": torch.tensor(dsts, dtype=torch.int64),
        "sizes": torch.tensor(sizes, dtype=torch.int32),
        "src_slots_k": src_slots_k,
        "src_slots_v": src_slots_v,
        "dst_slots_k": dst_slots_k,
        "dst_slots_v": dst_slots_v,
    }
    return host_pool_k, host_pool_v, topk_k, topk_v, desc


def test_packed_host_gather_and_fill_scatter_slots_cpu():
    cpp = _load_sparse_kv_offload_cpp()
    num_k, num_v = 4, 3
    token_k, token_v = 512, 128
    host_pool_k, host_pool_v, topk_k, topk_v, desc = _make_descriptors(num_k, num_v, token_k, token_v, 17)

    num_entries = num_k + num_v
    buffer_bytes = num_k * token_k + num_v * token_v
    packed = torch.zeros(buffer_bytes, dtype=torch.uint8, pin_memory=True)
    slots_k = torch.zeros(num_k, dtype=torch.int64)
    slots_v = torch.zeros(num_v, dtype=torch.int64)

    n_k, n_v, nbytes = cpp.packed_fill_scatter_slots(
        desc["src"],
        desc["dst"],
        desc["sizes"],
        num_entries,
        int(topk_k.data_ptr()),
        int(topk_v.data_ptr()),
        token_k,
        token_v,
        buffer_bytes,
        slots_k,
        slots_v,
    )
    assert (n_k, n_v, nbytes) == (num_k, num_v, buffer_bytes)
    assert slots_k.tolist() == desc["dst_slots_k"]
    assert slots_v.tolist() == desc["dst_slots_v"]

    packed_bytes = int(
        cpp.packed_host_gather(
            desc["src"],
            desc["dst"],
            desc["sizes"],
            num_entries,
            int(packed.data_ptr()),
            buffer_bytes,
            4,
        )
    )
    assert packed_bytes == buffer_bytes
    # Pack order must match the slot order: K rows first, then V rows.
    for row, src_slot in enumerate(desc["src_slots_k"]):
        got = packed[row * token_k : (row + 1) * token_k]
        assert torch.equal(got, host_pool_k[src_slot]), f"packed K row {row} mismatch"
    v_start = num_k * token_k
    for row, src_slot in enumerate(desc["src_slots_v"]):
        got = packed[v_start + row * token_v : v_start + (row + 1) * token_v]
        assert torch.equal(got, host_pool_v[src_slot]), f"packed V row {row} mismatch"


def test_packed_fill_scatter_slots_skips_invalid_and_detects_overflow():
    cpp = _load_sparse_kv_offload_cpp()
    num_k, num_v = 4, 3
    token_k, token_v = 512, 128
    _, _, topk_k, topk_v, desc = _make_descriptors(num_k, num_v, token_k, token_v, 3)

    # Invalidate one K entry (size=0) and one V entry (src=0).
    desc["sizes"][1] = 0
    desc["src"][num_k + 1] = 0
    num_entries = num_k + num_v
    buffer_bytes = num_k * token_k + num_v * token_v
    slots_k = torch.zeros(num_k, dtype=torch.int64)
    slots_v = torch.zeros(num_v, dtype=torch.int64)

    n_k, n_v, nbytes = cpp.packed_fill_scatter_slots(
        desc["src"],
        desc["dst"],
        desc["sizes"],
        num_entries,
        int(topk_k.data_ptr()),
        int(topk_v.data_ptr()),
        token_k,
        token_v,
        buffer_bytes,
        slots_k,
        slots_v,
    )
    expect_k = [desc["dst_slots_k"][0]] + desc["dst_slots_k"][2:]
    expect_v = [desc["dst_slots_v"][0], desc["dst_slots_v"][2]]
    assert (n_k, n_v) == (3, 2)
    assert nbytes == 3 * token_k + 2 * token_v
    assert slots_k[:n_k].tolist() == expect_k
    assert slots_v[:n_v].tolist() == expect_v

    # The pack must skip the same entries and stay consistent with the slots.
    packed = torch.zeros(buffer_bytes, dtype=torch.uint8, pin_memory=True)
    packed_bytes = int(
        cpp.packed_host_gather(
            desc["src"],
            desc["dst"],
            desc["sizes"],
            num_entries,
            int(packed.data_ptr()),
            buffer_bytes,
            4,
        )
    )
    assert packed_bytes == nbytes

    # Overflow: a single item larger than the buffer reports -1.
    n_k2, n_v2, nbytes2 = cpp.packed_fill_scatter_slots(
        desc["src"],
        desc["dst"],
        desc["sizes"],
        num_entries,
        int(topk_k.data_ptr()),
        int(topk_v.data_ptr()),
        token_k,
        token_v,
        16,
        slots_k,
        slots_v,
    )
    assert nbytes2 == -1


def test_packed_onload_end_to_end_without_memfabric():
    cpp = _load_sparse_kv_offload_cpp()
    device = torch.device("npu:0")
    torch.npu.set_device(0)

    num_k, num_v = 4, 3
    token_k, token_v = 512, 128
    host_pool_k, host_pool_v, topk_k, topk_v, desc = _make_descriptors(num_k, num_v, token_k, token_v, 29)

    num_entries = num_k + num_v
    buffer_bytes = num_k * token_k + num_v * token_v
    packed_host = torch.zeros(buffer_bytes, dtype=torch.uint8, pin_memory=True)
    slots_k_cpu = torch.zeros(num_k, dtype=torch.int64, pin_memory=True)
    slots_v_cpu = torch.zeros(num_v, dtype=torch.int64, pin_memory=True)

    n_k, n_v, nbytes = cpp.packed_fill_scatter_slots(
        desc["src"],
        desc["dst"],
        desc["sizes"],
        num_entries,
        int(topk_k.data_ptr()),
        int(topk_v.data_ptr()),
        token_k,
        token_v,
        buffer_bytes,
        slots_k_cpu,
        slots_v_cpu,
    )
    assert nbytes == buffer_bytes
    packed_bytes = int(
        cpp.packed_host_gather(
            desc["src"],
            desc["dst"],
            desc["sizes"],
            num_entries,
            int(packed_host.data_ptr()),
            buffer_bytes,
            4,
        )
    )
    assert packed_bytes == nbytes

    # One contiguous H2D, then D2D scatter via index_copy_. No memfabric.
    staging_npu = torch.zeros(buffer_bytes, dtype=torch.uint8, device=device)
    staging_npu[:nbytes].copy_(packed_host[:nbytes], non_blocking=True)
    slots_k_npu = torch.empty(num_k, dtype=torch.int64, device=device)
    slots_v_npu = torch.empty(num_v, dtype=torch.int64, device=device)
    slots_k_npu[:n_k].copy_(slots_k_cpu[:n_k], non_blocking=True)
    slots_v_npu[:n_v].copy_(slots_v_cpu[:n_v], non_blocking=True)

    staging_k = staging_npu[: n_k * token_k].view(torch.bfloat16).view(n_k, token_k // 2)
    topk_k.view(-1, token_k // 2).index_copy_(0, slots_k_npu[:n_k], staging_k)
    v_start = n_k * token_k
    staging_v = staging_npu[v_start : v_start + n_v * token_v].view(torch.bfloat16).view(n_v, token_v // 2)
    topk_v.view(-1, token_v // 2).index_copy_(0, slots_v_npu[:n_v], staging_v)
    torch.npu.synchronize()

    for row, (src_slot, dst_slot) in enumerate(zip(desc["src_slots_k"], desc["dst_slots_k"])):
        expect = host_pool_k[src_slot].view(torch.bfloat16)
        assert torch.equal(topk_k[dst_slot].cpu(), expect), f"K slot {dst_slot} mismatch"
    for row, (src_slot, dst_slot) in enumerate(zip(desc["src_slots_v"], desc["dst_slots_v"])):
        expect = host_pool_v[src_slot].view(torch.bfloat16)
        assert torch.equal(topk_v[dst_slot].cpu(), expect), f"V slot {dst_slot} mismatch"
