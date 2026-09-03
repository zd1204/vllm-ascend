# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

import ctypes
import types
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm_ascend.kv_offload.cpu_gather import (
    CpuGatherPool,
    GatherItem,
    build_gather_items,
    split_by_buffer_capacity,
)


def test_build_gather_items_layout():
    src_ptrs = [1000, 2000, 3000]
    sizes = [64, 128, 32]
    items, total = build_gather_items(src_ptrs, sizes)

    assert total == 224
    assert items == [
        GatherItem(src_ptr=1000, dst_offset=0, size=64),
        GatherItem(src_ptr=2000, dst_offset=64, size=128),
        GatherItem(src_ptr=3000, dst_offset=192, size=32),
    ]


def test_cpu_gather_pool_copies_discrete_blocks():
    block_size = 64
    num_blocks = 8
    src = (ctypes.c_uint8 * (num_blocks * block_size * 2))()
    dst = (ctypes.c_uint8 * (num_blocks * block_size))()

    items: list[GatherItem] = []
    for i in range(num_blocks):
        src_offset = i * block_size * 2
        for j in range(block_size):
            src[src_offset + j] = (i * 17 + j) & 0xFF
        items.append(
            GatherItem(
                src_ptr=ctypes.addressof(src) + src_offset,
                dst_offset=i * block_size,
                size=block_size,
            )
        )

    pool = CpuGatherPool(num_threads=4)
    try:
        pool.gather(items, ctypes.addressof(dst))
    finally:
        pool.close()

    for i in range(num_blocks):
        for j in range(block_size):
            assert dst[i * block_size + j] == (i * 17 + j) & 0xFF


def test_cpu_gather_pool_empty_is_noop():
    pool = CpuGatherPool(num_threads=2)
    try:
        pool.gather([], 0)
    finally:
        pool.close()


def test_split_by_buffer_capacity_chunks_and_fallback():
    sizes = np.array([100, 100, 100, 50], dtype=np.int64)
    assert split_by_buffer_capacity(sizes, buffer_bytes=250) == [(0, 2), (2, 4)]
    assert split_by_buffer_capacity(sizes, buffer_bytes=50) == []
    assert split_by_buffer_capacity(np.array([], dtype=np.int64), 100) == []


def test_cpu_gather_enabled_on_eager_and_graph():
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        SparseKVOffloadManager,
    )

    manager = MagicMock()
    manager._enable_cpu_gather_h2d = True
    manager._packed_gva = 0x1000
    manager.tp_rank = 0
    assert SparseKVOffloadManager._should_use_cpu_gather(manager, capturing=False) is True
    assert SparseKVOffloadManager._should_use_cpu_gather(manager, capturing=True) is True
    manager.tp_rank = 1
    assert SparseKVOffloadManager._should_use_cpu_gather(manager, capturing=False) is True
    manager._enable_cpu_gather_h2d = False
    manager.tp_rank = 0
    assert SparseKVOffloadManager._should_use_cpu_gather(manager, capturing=False) is False
    manager._enable_cpu_gather_h2d = True
    manager._packed_gva = 0
    assert SparseKVOffloadManager._should_use_cpu_gather(manager) is False


def test_tp0_pack_host_gather_only_copies_on_rank0():
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        SparseKVOffloadManager,
    )

    packed_calls: list[int] = []

    def fake_pack(*args):
        packed_calls.append(int(args[3]))
        return 64

    host_src = (ctypes.c_uint8 * 64)()
    manager = MagicMock()
    manager._packed_gva = ctypes.addressof(host_src)
    manager._cpu_gather_buffer_bytes = 4096
    manager._cpu_gather_threads = 4
    manager.tp_rank = 0
    manager.num_tokens_buffer_cpu = torch.tensor([2], dtype=torch.int32)
    manager.gvas_buffer_cpu = torch.tensor(
        [ctypes.addressof(host_src), ctypes.addressof(host_src) + 32],
        dtype=torch.int64,
    )
    manager.addr_buffer_cpu = torch.tensor([1000, 2000], dtype=torch.int64)
    manager.size_buffer_cpu = torch.tensor([32, 32], dtype=torch.int32)
    manager.sparse_kv_offload_cpp = types.SimpleNamespace(packed_host_gather=fake_pack)

    assert SparseKVOffloadManager._tp0_pack_host_gather(manager) is True
    assert packed_calls == [2]
    assert manager._packed_nbytes == 64

    packed_calls.clear()
    manager.tp_rank = 1
    assert SparseKVOffloadManager._tp0_pack_host_gather(manager) is True
    assert packed_calls == []
    assert manager._packed_nbytes == 64


def test_prepare_packed_onload_cpu_rewrites_gva_srcs():
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        SparseKVOffloadManager,
    )

    packed_calls: list[int] = []
    fill_calls: list[int] = []
    host_src = (ctypes.c_uint8 * 64)()
    packed_gva = ctypes.addressof(host_src)

    def fake_pack(*args):
        packed_calls.append(int(args[3]))
        return 64

    def fake_fill(*args):
        fill_calls.append(int(args[3]))
        scatter = args[6]
        scatter[0] = packed_gva
        scatter[1] = packed_gva + 32
        return True

    manager = MagicMock()
    manager._packed_gva = packed_gva
    manager._packed_nbytes = 0
    manager._cpu_gather_buffer_bytes = 4096
    manager._cpu_gather_threads = 4
    manager.tp_rank = 0
    manager.num_tokens_buffer_cpu = torch.tensor([2], dtype=torch.int32)
    manager.gvas_buffer_cpu = torch.tensor(
        [packed_gva + 100, packed_gva + 200],
        dtype=torch.int64,
    )
    manager.addr_buffer_cpu = torch.tensor([1000, 2000], dtype=torch.int64)
    manager.size_buffer_cpu = torch.tensor([32, 32], dtype=torch.int32)
    manager.sparse_kv_offload_cpp = types.SimpleNamespace(
        packed_host_gather=fake_pack,
        packed_fill_scatter_srcs=fake_fill,
    )
    manager._tp0_pack_host_gather = lambda: SparseKVOffloadManager._tp0_pack_host_gather(manager)
    manager._disable_packed_transfer = lambda: SparseKVOffloadManager._disable_packed_transfer(manager)

    SparseKVOffloadManager._prepare_packed_onload_cpu(manager)
    assert packed_calls == [2]
    assert fill_calls == [2]
    assert manager._packed_nbytes == 64
    assert manager.gvas_buffer_cpu.tolist() == [packed_gva, packed_gva + 32]
    assert manager.addr_buffer_cpu.tolist() == [1000, 2000]

    packed_calls.clear()
    fill_calls.clear()
    manager.tp_rank = 1
    SparseKVOffloadManager._prepare_packed_onload_cpu(manager)
    assert packed_calls == []
    assert fill_calls == [2]


def test_onload_topk_kv_cpu_runs_pack_after_lru():
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        SparseKVOffloadManager,
    )

    order: list[str] = []

    manager = MagicMock()
    manager.topk = 2
    manager.topk_buffer_size = 4
    manager.max_model_len = 8
    manager.lru_workspace_threads = 1
    manager._enable_cpu_gather_h2d = True
    manager._packed_gva = 0x1000
    manager.sparse_kv_offload_cpp = types.SimpleNamespace(
        lru_resident_compact=lambda *args: order.append("lru"),
        compute_lru_resident_addrs=lambda *args: order.append("addrs"),
    )
    manager._prepare_packed_onload_cpu = lambda: order.append("pack")
    manager._should_use_cpu_gather = lambda *args, **kwargs: SparseKVOffloadManager._should_use_cpu_gather(manager)

    dummy = 0
    args = (1,) + (dummy,) * 31
    SparseKVOffloadManager._onload_topk_kv_cpu(manager, args)
    assert order == ["lru", "addrs", "pack"]


def test_tp0_pack_host_gather_falls_back_when_too_large():
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        SparseKVOffloadManager,
    )

    manager = MagicMock()
    manager._packed_gva = 1
    manager._cpu_gather_buffer_bytes = 16
    manager.tp_rank = 0
    manager.num_tokens_buffer_cpu = torch.tensor([1], dtype=torch.int32)
    manager.gvas_buffer_cpu = torch.tensor([1], dtype=torch.int64)
    manager.addr_buffer_cpu = torch.tensor([2], dtype=torch.int64)
    manager.size_buffer_cpu = torch.tensor([64], dtype=torch.int32)
    manager.sparse_kv_offload_cpp = types.SimpleNamespace(packed_host_gather=lambda *args: -1)

    assert SparseKVOffloadManager._tp0_pack_host_gather(manager) is False


def test_prepare_packed_onload_cpu_overflow_is_noop():
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        SparseKVOffloadManager,
    )

    packed_calls: list[int] = []
    manager = MagicMock()
    manager._packed_gva = 1
    manager._packed_nbytes = 99
    manager._cpu_gather_buffer_bytes = 16
    manager._cpu_gather_threads = 4
    manager.tp_rank = 0
    manager.num_tokens_buffer_cpu = torch.tensor([2], dtype=torch.int32)
    manager.gvas_buffer_cpu = torch.tensor([1, 2], dtype=torch.int64)
    manager.addr_buffer_cpu = torch.tensor([1000, 2000], dtype=torch.int64)
    manager.size_buffer_cpu = torch.tensor([64, 64], dtype=torch.int32)
    manager.sparse_kv_offload_cpp = types.SimpleNamespace(
        packed_host_gather=lambda *args: packed_calls.append(1) or -1,
        packed_fill_scatter_srcs=lambda *args: True,
    )
    manager._tp0_pack_host_gather = lambda: SparseKVOffloadManager._tp0_pack_host_gather(manager)
    manager._disable_packed_transfer = lambda: SparseKVOffloadManager._disable_packed_transfer(manager)

    SparseKVOffloadManager._prepare_packed_onload_cpu(manager)
    assert packed_calls == []
    assert manager._packed_nbytes == 0
    assert manager.size_buffer_cpu.tolist() == [0, 0]
