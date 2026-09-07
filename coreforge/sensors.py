"""Единый слой датчиков: NVML на NVIDIA, ADL на AMD."""

from __future__ import annotations

from coreforge.nvml_api import GpuSample, Nvml
from coreforge.vendor import GpuCaps


class Sensors:
    def __init__(self, caps: GpuCaps | None = None) -> None:
        self.nvml = None
        self.adl = None
        self.available = False
        if caps is None or caps.is_nvidia or caps.has_nvml:
            self.nvml = Nvml()
            self.available = bool(self.nvml.available)
        if not self.available:
            try:
                from coreforge.adl_api import Adl

                self.adl = Adl()
                self.available = bool(self.adl.available)
            except Exception:
                self.adl = None

    def sample(self) -> GpuSample:
        if self.nvml and self.nvml.available:
            return self.nvml.sample()
        if self.adl and self.adl.available:
            return self.adl.sample()
        return GpuSample(error="нет NVML/ADL")

    def close(self) -> None:
        if self.nvml:
            self.nvml.close()
        if self.adl:
            self.adl.close()
