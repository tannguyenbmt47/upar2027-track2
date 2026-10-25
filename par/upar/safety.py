"""Protect a laptop during long runs: pause when the CPU is too hot, stop when RAM runs out."""
import time
from pathlib import Path


def cpu_temp():
    """AMD k10temp / Intel coretemp package temperature in C, 0 if unknown (e.g. on Colab)."""
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            name = (h / "name").read_text().strip()
            if name in ("k10temp", "coretemp"):
                return int((h / "temp1_input").read_text()) / 1000
        except OSError:
            continue
    return 0.0


def ram_free_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024
    return float("inf")


def guard(max_temp=103, resume_temp=95, min_ram_mb=1500, max_wait_s=300):
    if ram_free_mb() < min_ram_mb:
        raise RuntimeError(f"free RAM {ram_free_mb():.0f} MB < {min_ram_mb} MB: stopping to protect the machine "
                           f"(rerun with --resume to continue from the last epoch)")
    if cpu_temp() >= max_temp:
        print(f"  [guard] CPU {cpu_temp():.0f}C >= {max_temp}C, pausing", flush=True)
        t0 = time.time()
        while cpu_temp() > resume_temp and time.time() - t0 < max_wait_s:
            time.sleep(5)
        print(f"  [guard] resumed at {cpu_temp():.0f}C after {time.time() - t0:.0f}s", flush=True)
