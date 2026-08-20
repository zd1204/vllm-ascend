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
"""Multithreaded host gather for discrete CPU blocks into a contiguous buffer."""

from __future__ import annotations

import ctypes
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class GatherItem:
    """One memcpy from a discrete source into a contiguous gather buffer."""

    src_ptr: int
    dst_offset: int
    size: int


def _memmove(dst_ptr: int, src_ptr: int, size: int) -> None:
    if size <= 0:
        return
    ctypes.memmove(dst_ptr, src_ptr, size)


class CpuGatherPool:
    """Reusable worker pool that gathers discrete host blocks via memcpy.

    Workers are started once and woken per batch (generation barrier), matching
    the long-lived thread pool pattern used by the DDR gather C-scheme benches.
    """

    def __init__(self, num_threads: int = 4):
        self._num_threads = max(1, num_threads)
        self._mu = threading.Lock()
        self._cv_start = threading.Condition(self._mu)
        self._cv_done = threading.Condition(self._mu)
        self._cv_started = threading.Condition(self._mu)

        self._stop = False
        self._generation = 0
        self._started_workers = 0
        self._completed_workers = 0
        self._items: list[GatherItem] | None = None
        self._gather_base: int = 0

        self._threads = [
            threading.Thread(
                target=self._worker_loop,
                name=f"cpu-gather-{i}",
                args=(i,),
                daemon=True,
            )
            for i in range(self._num_threads)
        ]
        for thread in self._threads:
            thread.start()

        with self._cv_started:
            self._cv_started.wait_for(lambda: self._started_workers == self._num_threads)

    def close(self) -> None:
        with self._mu:
            self._stop = True
            self._generation += 1
            self._cv_start.notify_all()
        for thread in self._threads:
            thread.join(timeout=5.0)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def gather(self, items: list[GatherItem], gather_base_ptr: int) -> None:
        """Copy all items into the contiguous buffer starting at gather_base_ptr."""
        if not items:
            return
        if self._num_threads == 1:
            for item in items:
                _memmove(gather_base_ptr + item.dst_offset, item.src_ptr, item.size)
            return

        with self._mu:
            self._items = items
            self._gather_base = gather_base_ptr
            self._completed_workers = 0
            self._generation += 1
            self._cv_start.notify_all()
            self._cv_done.wait_for(lambda: self._completed_workers == self._num_threads)
            self._items = None

    def _worker_loop(self, worker_index: int) -> None:
        seen_generation = 0
        with self._cv_started:
            self._started_workers += 1
            if self._started_workers == self._num_threads:
                self._cv_started.notify_all()

        while True:
            with self._cv_start:
                self._cv_start.wait_for(lambda: self._stop or self._generation != seen_generation)
                if self._stop:
                    return
                seen_generation = self._generation
                items = self._items
                gather_base = self._gather_base
                assert items is not None
                begin = len(items) * worker_index // self._num_threads
                end = len(items) * (worker_index + 1) // self._num_threads

            for item in items[begin:end]:
                _memmove(gather_base + item.dst_offset, item.src_ptr, item.size)

            with self._cv_done:
                self._completed_workers += 1
                if self._completed_workers == self._num_threads:
                    self._cv_done.notify_all()


def build_gather_items(src_ptrs: list[int] | tuple[int, ...], sizes: list[int] | tuple[int, ...]) -> tuple[list[GatherItem], int]:
    """Build gather items with packed contiguous destination offsets.

    Returns:
        (items, total_bytes)
    """
    if len(src_ptrs) != len(sizes):
        raise ValueError("src_ptrs and sizes must have the same length")
    items: list[GatherItem] = []
    offset = 0
    for src_ptr, size in zip(src_ptrs, sizes):
        size_i = int(size)
        if size_i < 0:
            raise ValueError(f"negative gather size: {size_i}")
        items.append(GatherItem(src_ptr=int(src_ptr), dst_offset=offset, size=size_i))
        offset += size_i
    return items, offset


def split_by_buffer_capacity(
    sizes: list[int] | tuple[int, ...] | "np.ndarray",
    buffer_bytes: int,
) -> list[tuple[int, int]]:
    """Split item index range into chunks that fit into buffer_bytes.

    Returns list of [begin, end) index ranges. A single item larger than
    buffer_bytes yields an empty list (caller should fall back).
    """
    import numpy as np

    sizes_arr = np.asarray(sizes, dtype=np.int64)
    if sizes_arr.size == 0:
        return []
    if int(sizes_arr.max()) > buffer_bytes:
        return []

    ranges: list[tuple[int, int]] = []
    begin = 0
    used = 0
    for idx, size in enumerate(sizes_arr.tolist()):
        size_i = int(size)
        if used > 0 and used + size_i > buffer_bytes:
            ranges.append((begin, idx))
            begin = idx
            used = 0
        used += size_i
    ranges.append((begin, sizes_arr.size))
    return ranges
