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
"""Unit tests for the packed onload path (CPU gather + one contiguous H2D +
D2D index_copy_ scatter). These tests mock the cpp extension and the manager
state, so they run without NPU or memfabric."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
)


def _make_manager(**overrides) -> MagicMock:
    manager = MagicMock()
    manager._enable_cpu_gather_h2d = True
    manager.tp_rank = 0
    manager._packed_nbytes = 0
    manager._packed_n_k = 0
    manager._packed_n_v = 0
    manager._cpu_gather_buffer_bytes = 4096
    manager._cpu_gather_threads = 4
    for key, value in overrides.items():
        setattr(manager, key, value)
    return manager


def test_should_use_cpu_gather_eager_only():
    manager = _make_manager()
    assert SparseKVOffloadManager._should_use_cpu_gather(manager, capturing=False) is True
    # Graph capture/replay keeps the discrete memfabric path.
    assert SparseKVOffloadManager._should_use_cpu_gather(manager, capturing=True) is False
    manager._enable_cpu_gather_h2d = False
    assert SparseKVOffloadManager._should_use_cpu_gather(manager, capturing=False) is False


def test_prepare_packed_onload_cpu_packs_on_tp0_only():
    slots_calls: list[int] = []
    pack_calls: list[int] = []

    def fake_slots(*args):
        slots_calls.append(int(args[3]))
        return (2, 1, 96)

    def fake_pack(*args):
        pack_calls.append(int(args[3]))
        return 96

    manager = _make_manager(
        num_tokens_buffer_cpu=torch.tensor([3], dtype=torch.int32),
        gvas_buffer_cpu=torch.zeros(4, dtype=torch.int64),
        addr_buffer_cpu=torch.zeros(4, dtype=torch.int64),
        size_buffer_cpu=torch.zeros(4, dtype=torch.int32),
        _scatter_slots_k_cpu=torch.zeros(4, dtype=torch.int64),
        _scatter_slots_v_cpu=torch.zeros(4, dtype=torch.int64),
        _packed_host_buf=torch.zeros(4096, dtype=torch.int8),
        sparse_kv_offload_cpp=types.SimpleNamespace(
            packed_fill_scatter_slots=fake_slots,
            packed_host_gather=fake_pack,
        ),
    )

    SparseKVOffloadManager._prepare_packed_onload_cpu(
        manager,
        manager.gvas_buffer_cpu,
        manager.addr_buffer_cpu,
        manager.size_buffer_cpu,
        manager.num_tokens_buffer_cpu,
        1000,
        2000,
        32,
        32,
    )
    assert slots_calls == [3]
    assert pack_calls == [3]
    assert (manager._packed_n_k, manager._packed_n_v, manager._packed_nbytes) == (2, 1, 96)

    # Non-TP0 ranks build slot metadata but never pack host data.
    slots_calls.clear()
    pack_calls.clear()
    manager.tp_rank = 1
    SparseKVOffloadManager._prepare_packed_onload_cpu(
        manager,
        manager.gvas_buffer_cpu,
        manager.addr_buffer_cpu,
        manager.size_buffer_cpu,
        manager.num_tokens_buffer_cpu,
        1000,
        2000,
        32,
        32,
    )
    assert slots_calls == [3]
    assert pack_calls == []
    assert manager._packed_nbytes == 96


def test_prepare_packed_onload_cpu_empty_entries_is_noop():
    manager = _make_manager(
        num_tokens_buffer_cpu=torch.tensor([0], dtype=torch.int32),
        gvas_buffer_cpu=torch.zeros(4, dtype=torch.int64),
        sparse_kv_offload_cpp=types.SimpleNamespace(
            packed_fill_scatter_slots=lambda *args: (_ for _ in ()).throw(AssertionError("must not be called")),
        ),
    )
    manager._packed_nbytes = 99
    SparseKVOffloadManager._prepare_packed_onload_cpu(
        manager,
        manager.gvas_buffer_cpu,
        MagicMock(),
        MagicMock(),
        manager.num_tokens_buffer_cpu,
        1000,
        2000,
        32,
        32,
    )
    assert manager._packed_nbytes == 0


def test_prepare_packed_onload_cpu_overflow_falls_back():
    def fake_slots(*args):
        return (1, 1, -1)

    pack_calls: list[int] = []
    manager = _make_manager(
        num_tokens_buffer_cpu=torch.tensor([2], dtype=torch.int32),
        gvas_buffer_cpu=torch.zeros(4, dtype=torch.int64),
        addr_buffer_cpu=torch.zeros(4, dtype=torch.int64),
        size_buffer_cpu=torch.zeros(4, dtype=torch.int32),
        _scatter_slots_k_cpu=torch.zeros(4, dtype=torch.int64),
        _scatter_slots_v_cpu=torch.zeros(4, dtype=torch.int64),
        sparse_kv_offload_cpp=types.SimpleNamespace(
            packed_fill_scatter_slots=fake_slots,
            packed_host_gather=lambda *args: pack_calls.append(1) or -1,
        ),
    )
    manager._packed_nbytes = 99
    SparseKVOffloadManager._prepare_packed_onload_cpu(
        manager,
        manager.gvas_buffer_cpu,
        manager.addr_buffer_cpu,
        manager.size_buffer_cpu,
        manager.num_tokens_buffer_cpu,
        1000,
        2000,
        32,
        32,
    )
    # Overflow: no pack attempt, nbytes reset so the caller falls back.
    assert pack_calls == []
    assert manager._packed_nbytes == 0


def test_prepare_packed_onload_cpu_pack_mismatch_falls_back():
    manager = _make_manager(
        num_tokens_buffer_cpu=torch.tensor([2], dtype=torch.int32),
        gvas_buffer_cpu=torch.zeros(4, dtype=torch.int64),
        addr_buffer_cpu=torch.zeros(4, dtype=torch.int64),
        size_buffer_cpu=torch.zeros(4, dtype=torch.int32),
        _scatter_slots_k_cpu=torch.zeros(4, dtype=torch.int64),
        _scatter_slots_v_cpu=torch.zeros(4, dtype=torch.int64),
        _packed_host_buf=torch.zeros(4096, dtype=torch.int8),
        sparse_kv_offload_cpp=types.SimpleNamespace(
            packed_fill_scatter_slots=lambda *args: (1, 1, 64),
            packed_host_gather=lambda *args: 32,  # mismatch with expected 64
        ),
    )
    SparseKVOffloadManager._prepare_packed_onload_cpu(
        manager,
        manager.gvas_buffer_cpu,
        manager.addr_buffer_cpu,
        manager.size_buffer_cpu,
        manager.num_tokens_buffer_cpu,
        1000,
        2000,
        32,
        32,
    )
    assert manager._packed_nbytes == 0


def test_onload_topk_kv_cpu_runs_pack_after_lru_only_when_enabled():
    order: list[str] = []

    def _build_manager() -> MagicMock:
        manager = MagicMock()
        manager.topk = 2
        manager.topk_buffer_size = 4
        manager.max_model_len = 8
        manager.lru_workspace_threads = 1
        manager.sparse_kv_offload_cpp = types.SimpleNamespace(
            lru_resident_compact=lambda *args: order.append("lru"),
            compute_lru_resident_addrs=lambda *args: order.append("addrs"),
        )
        manager._prepare_packed_onload_cpu = lambda *args: order.append("pack")
        return manager

    dummy = 0
    base_args = (1,) + (dummy,) * 30  # 31 entries before layer_id

    manager = _build_manager()
    SparseKVOffloadManager._onload_topk_kv_cpu(manager, base_args + (0, True))
    assert order == ["lru", "addrs", "pack"]

    order.clear()
    manager = _build_manager()
    SparseKVOffloadManager._onload_topk_kv_cpu(manager, base_args + (0, False))
    assert order == ["lru", "addrs"]


def test_scatter_packed_onload_index_copy_layout():
    """Verify the packed staging layout: K rows first, then V rows, scattered
    into topk buffers by slot index. Runs on CPU tensors (index_copy_ semantics
    are device-agnostic)."""
    token_k = 8  # bytes, 4 bf16
    token_v = 4  # bytes, 2 bf16
    n_k, n_v = 3, 2
    num_slots_k, num_slots_v = 8, 4

    manager = _make_manager()
    manager._packed_n_k = n_k
    manager._packed_n_v = n_v
    manager.token_size_bytes_k = token_k
    manager.token_size_bytes_v = token_v

    staging = torch.zeros(n_k * token_k + n_v * token_v, dtype=torch.int8)
    for i in range(staging.numel()):
        staging[i] = (i * 7 + 1) & 0x7F
    manager._packed_staging_npu = staging

    slots_k = torch.tensor([5, 1, 3], dtype=torch.int64)
    slots_v = torch.tensor([2, 0], dtype=torch.int64)
    manager._scatter_slots_k_npu = slots_k
    manager._scatter_slots_v_npu = slots_v

    topk_k = torch.zeros(num_slots_k, token_k // 2, dtype=torch.bfloat16)
    topk_v = torch.zeros(num_slots_v, token_v // 2, dtype=torch.bfloat16)
    manager.topk_buffers_k = [topk_k.view(num_slots_k, 1, 1, token_k // 2)]
    manager.topk_buffers_v = [topk_v.view(num_slots_v, 1, 1, token_v // 2)]

    SparseKVOffloadManager._scatter_packed_onload(manager, layer_id=0)

    staging_k = staging[: n_k * token_k].view(torch.bfloat16).view(n_k, token_k // 2)
    staging_v = staging[n_k * token_k :].view(torch.bfloat16).view(n_v, token_v // 2)
    for row, slot in enumerate(slots_k.tolist()):
        assert torch.equal(topk_k[slot], staging_k[row]), f"K slot {slot} mismatch"
    for row, slot in enumerate(slots_v.tolist()):
        assert torch.equal(topk_v[slot], staging_v[row]), f"V slot {slot} mismatch"
    # Untouched slots stay zero.
    assert topk_k[0].abs().sum().item() == 0
    assert topk_v[1].abs().sum().item() == 0
