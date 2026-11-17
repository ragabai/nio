"""The machine and the packages a run ran on, to keep beside its results."""

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Self

PACKAGES = ("numpy", "plotly", "kaleido", "nio")  # the versions worth recording, when installed


@dataclass(frozen=True)
class System:
    """The operating system, machine, processor, cores, Python and package versions of this computer."""

    os: str
    machine: str
    cpu: str
    cores: int
    python: str
    packages: dict[str, str]

    @classmethod
    def inspect(cls) -> Self:
        """This computer, as it is now."""
        packages = {}
        for name in PACKAGES:
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
        return cls(
            os=f"{platform.system()} {platform.release()}",
            machine=platform.machine() or "unknown",
            cpu=_cpu(),
            cores=os.cpu_count() or 1,
            python=platform.python_version(),
            packages=packages,
        )

    def save(self, path) -> Path:
        """Write it as JSON, creating the folder if needed, and return the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return out

    def __str__(self) -> str:
        packages = ", ".join(f"{name} {version}" for name, version in self.packages.items())
        return f"{self.os} ({self.machine}), {self.cpu}, {self.cores} cores\nPython {self.python}; {packages}"


def _cpu() -> str:
    """The processor's name as its maker gives it, where the system records one (Windows registry, Linux
    /proc/cpuinfo); otherwise what `platform` knows."""
    if sys.platform == "win32":
        try:
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as cpu:
                return str(winreg.QueryValueEx(cpu, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text("utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "unknown"
