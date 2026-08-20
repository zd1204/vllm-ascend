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


def test_cpu_gather_enabled_on_all_eager_ranks():
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        SparseKVOffloadManager,
    )

    manager = MagicMock()
    manager._enable_cpu_gather_h2d = True
    manager._packed_gva = 0x1000
    manager.tp_rank = 0
    assert SparseKVOffloadManager._should_use_cpu_gather(manager, capturing=False) is True
    assert SparseKVOffloadManager._should_use_cpu_gather(manager, capturing=True) is False
    manager.tp_rank = 1
    assert SparseKVOffloadManager._should_use_cpu_gather(manager, capturing=False) is True
    manager._enable_cpu_gather_h2d = False
    manager.tp_rank = 0
    assert SparseKVOffloadManager._should_use_cpu_gather(manager, capturing=False) is False


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
    manager._packed_nbytes = 0
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


def test_submit_packed_h2d_tp0_uses_contiguous_acl(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        SparseKVOffloadManager,
    )

    h2d_calls: list[tuple] = []
    scatter_calls: list[int] = []

    manager = MagicMock()
    manager.tp_rank = 0
    manager._packed_gva = 0xABC0
    manager._packed_nbytes = 64
    manager._cpu_gather_buffer_bytes = 4096
    manager.topk_buffers_k = [torch.empty(1)]
    manager.num_tokens_buffer_cpu = torch.tensor([2], dtype=torch.int32)
    manager.gvas_buffer_cpu = torch.tensor([1, 2], dtype=torch.int64)
    manager.addr_buffer_cpu = torch.tensor([1000, 2000], dtype=torch.int64)
    manager.size_buffer_cpu = torch.tensor([32, 32], dtype=torch.int32)
    manager._device_staging_buf = None
    manager._packed_h2d_src_npu = None
    manager._ensure_cpu_gather_resources = lambda: SparseKVOffloadManager._ensure_cpu_gather_resources(manager)
    manager.sparse_kv_offload_cpp = types.SimpleNamespace(
        packed_contiguous_h2d=lambda *args: h2d_calls.append((int(args[0]), int(args[2]))) or True,
        packed_d2d_scatter=lambda *args: scatter_calls.append(int(args[3])) or True,
    )

    ok = SparseKVOffloadManager._submit_packed_sparse_h2d_and_d2d(manager)
    assert ok is True
    assert h2d_calls == [(0xABC0, 64)]
    assert scatter_calls == [2]


def test_submit_packed_h2d_non_tp0_uses_sparse_copy(monkeypatch):
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import sparse_kv_offload_manager as mod
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
        SparseKVOffloadManager,
    )

    sparse_calls: list[tuple] = []
    scatter_calls: list[int] = []
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None, raising=False)
    monkeypatch.setattr(
        mod.offload,
        "sparse_copy",
        lambda src, dst, sizes, num, device: sparse_calls.append(
            (int(src[0].item()), int(dst[0].item()), int(sizes[0].item()), int(num[0].item()))
        )
        or 0,
        raising=False,
    )

    manager = MagicMock()
    manager.tp_rank = 1
    manager._packed_gva = 0xABC0
    manager._packed_nbytes = 64
    manager._cpu_gather_buffer_bytes = 4096
    manager.topk_buffers_k = [torch.empty(1)]
    manager.num_tokens_buffer_cpu = torch.tensor([2], dtype=torch.int32)
    manager.gvas_buffer_cpu = torch.tensor([1, 2], dtype=torch.int64)
    manager.addr_buffer_cpu = torch.tensor([1000, 2000], dtype=torch.int64)
    manager.size_buffer_cpu = torch.tensor([32, 32], dtype=torch.int32)
    manager._device_staging_buf = None
    manager._packed_h2d_src_npu = None
    manager._ensure_cpu_gather_resources = lambda: SparseKVOffloadManager._ensure_cpu_gather_resources(manager)
    manager.sparse_kv_offload_cpp = types.SimpleNamespace(
        packed_d2d_scatter=lambda *args: scatter_calls.append(int(args[3])) or True
    )

    ok = SparseKVOffloadManager._submit_packed_sparse_h2d_and_d2d(manager)
    assert ok is True
    assert len(sparse_calls) == 1
    src, dst, nbytes, n = sparse_calls[0]
    assert src == 0xABC0
    assert nbytes == 64
    assert n == 1
    assert dst != 0
    assert scatter_calls == [2]


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
    manager.sparse_kv_offload_cpp = types.SimpleNamespace(packed_host_gather=lambda *args: 0)

    assert SparseKVOffloadManager._tp0_pack_host_gather(manager) is False
