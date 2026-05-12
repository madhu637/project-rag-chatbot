"""Lightweight system monitoring helpers (CPU / RAM)."""
from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass
class SystemStats:
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float

    @property
    def ram_display(self) -> str:
        return f"{self.ram_used_gb:.1f} / {self.ram_total_gb:.1f} GB ({self.ram_percent:.0f}%)"


def get_system_stats() -> SystemStats:
    cpu = psutil.cpu_percent(interval=0.1)
    vm = psutil.virtual_memory()
    return SystemStats(
        cpu_percent=cpu,
        ram_used_gb=vm.used / (1024 ** 3),
        ram_total_gb=vm.total / (1024 ** 3),
        ram_percent=vm.percent,
    )
import psutil

def get_system_stats():
    return {
        "cpu": psutil.cpu_percent(),
        "memory": psutil.virtual_memory().percent
    }