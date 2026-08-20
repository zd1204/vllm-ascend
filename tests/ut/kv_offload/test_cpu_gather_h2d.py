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


def test_sparse_manager_submit_h2d_cpu_gather_path(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        SparseKVOffloadManager,
    )

    packed_calls: list[tuple] = []

    manager = MagicMock()
    manager._gather_pool = None
    manager._host_gather_buf = None
    manager._device_staging_buf = None
    manager._cpu_gather_buffer_bytes = 4096
    manager._cpu_gather_threads = 1
    manager.topk_buffers_k = [torch.empty(1)]
    manager.num_tokens_buffer_cpu = torch.tensor([2], dtype=torch.int32)
    host_src = (ctypes.c_uint8 * 64)()
    manager.gvas_buffer_cpu = torch.tensor(
        [ctypes.addressof(host_src), ctypes.addressof(host_src) + 32],
        dtype=torch.int64,
    )
    manager.addr_buffer_cpu = torch.tensor([1000, 2000], dtype=torch.int64)
    manager.size_buffer_cpu = torch.tensor([32, 32], dtype=torch.int32)
    manager._gather_src_ptrs = None
    manager._gather_dst_ptrs = None
    manager._gather_sizes = None
    manager._gather_h2d_src = None
    manager._gather_h2d_dst = None
    manager._gather_h2d_size = None
    manager.sparse_kv_offload_cpp = types.SimpleNamespace(
        packed_h2d_d2d=lambda *args: packed_calls.append((args[2], args[3].tolist(), args[4].tolist()))
    )
    manager._ensure_cpu_gather_resources = lambda: SparseKVOffloadManager._ensure_cpu_gather_resources(manager)

    try:
        ok = SparseKVOffloadManager._submit_h2d_cpu_gather(manager)
        assert ok is True
        assert len(packed_calls) == 1
        nbytes, dsts, sizes = packed_calls[0]
        assert nbytes == 64
        assert dsts == [1000, 2000]
        assert sizes == [32, 32]
    finally:
        if manager._gather_pool is not None:
            manager._gather_pool.close()


def test_sparse_manager_submit_h2d_cpu_gather_falls_back_when_too_large():
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        SparseKVOffloadManager,
    )

    manager = MagicMock()
    manager._gather_pool = CpuGatherPool(num_threads=1)
    manager._host_gather_buf = torch.empty(16, dtype=torch.uint8, device="cpu")
    manager._device_staging_buf = torch.empty(16, dtype=torch.uint8, device="cpu")
    manager._cpu_gather_buffer_bytes = 16
    manager.num_tokens_buffer_cpu = torch.tensor([1], dtype=torch.int32)
    manager.gvas_buffer_cpu = torch.tensor([1], dtype=torch.int64)
    manager.addr_buffer_cpu = torch.tensor([2], dtype=torch.int64)
    manager.size_buffer_cpu = torch.tensor([64], dtype=torch.int32)
    manager._gather_dst_ptrs = torch.empty(1, dtype=torch.int64)
    manager._gather_sizes = torch.empty(1, dtype=torch.int64)
    manager.sparse_kv_offload_cpp = types.SimpleNamespace(packed_h2d_d2d=lambda *args: None)
    manager._ensure_cpu_gather_resources = lambda: True

    try:
        assert SparseKVOffloadManager._submit_h2d_cpu_gather(manager) is False
    finally:
        manager._gather_pool.close()
