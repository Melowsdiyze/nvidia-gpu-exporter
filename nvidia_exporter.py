#!/usr/bin/env python3
"""
NVIDIA SMI Prometheus Exporter
Exports NVIDIA GPU metrics for scraping by Prometheus.

Environment variables:
    NVIDIA_EXPORTER_PORT    — HTTP port (default: 9835)
    NVIDIA_SCRAPE_INTERVAL  — Scrape interval in seconds (default: 15)
    NVIDIA_LOG_DIR          — Log directory (default: logs/)
"""

import os
import pwd
import sys
import time
import logging
import logging.handlers
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple, Any, Union

from prometheus_client import (
    Gauge,
    Counter,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

EXPORTER_PORT = int(os.environ.get("NVIDIA_EXPORTER_PORT", "9835"))
SCRAPE_INTERVAL = int(os.environ.get("NVIDIA_SCRAPE_INTERVAL", "15"))
LOG_DIR = os.environ.get("NVIDIA_LOG_DIR", "logs")

# ---------------------------------------------------------------------------
# Log rotation — custom handler that removes the backup after rotation
# so total disk usage never exceeds MAX_LOG_BYTES.
# ---------------------------------------------------------------------------

LOG_MAX_BYTES = 150 * 1024 * 1024  # 150 MB
LOG_BACKUP_COUNT = 1


class TrimmedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    RotatingFileHandler subclass that immediately deletes the single backup
    file produced after rotation, keeping total log storage at or below
    LOG_MAX_BYTES at all times.
    """

    def doRollover(self) -> None:
        super().doRollover()
        # After rotation the old log is renamed to <logfile>.1 — delete it.
        backup_path = self.baseFilename + ".1"
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
        except OSError as exc:
            # Non-fatal: warn but continue
            self.handleError(None)  # type: ignore[arg-type]
            logging.getLogger(__name__).warning(
                "Could not remove backup log file %s: %s", backup_path, exc
            )


def _setup_logging() -> logging.Logger:
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "nvidia_exporter.log"

    handler = TrimmedRotatingFileHandler(
        filename=str(log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    logger = logging.getLogger("nvidia_exporter")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(stream_handler)
    return logger


logger = _setup_logging()

# ---------------------------------------------------------------------------
# Helper: run nvidia-smi safely
# ---------------------------------------------------------------------------

def _run_cmd(args: List[str], timeout: int = 30) -> Optional[str]:
    """
    Run a command and return stdout as a string, or None on failure.
    Stderr is silently discarded.
    """
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if result.returncode != 0:
            logger.debug(
                "Command %s exited with code %d: %s",
                args,
                result.returncode,
                result.stderr.decode(errors="replace").strip(),
            )
            return None
        return result.stdout.decode(errors="replace")
    except FileNotFoundError:
        logger.debug("Executable not found: %s", args[0])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out: %s", args)
        return None
    except Exception as exc:
        logger.debug("Unexpected error running %s: %s", args, exc)
        return None


def _nvidia_smi_available() -> bool:
    """Return True if nvidia-smi is present and responds."""
    out = _run_cmd(["nvidia-smi", "-L"])
    return out is not None


_cuda_version_cache: Optional[str] = None

def _get_cuda_version() -> str:
    """Parse CUDA version from nvidia-smi main output (cached)."""
    global _cuda_version_cache
    if _cuda_version_cache is not None:
        return _cuda_version_cache
    import re
    out = _run_cmd(["nvidia-smi"])
    if out:
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if m:
            _cuda_version_cache = m.group(1)
            return _cuda_version_cache
    _cuda_version_cache = "unknown"
    return _cuda_version_cache


# ---------------------------------------------------------------------------
# Value parsers
# ---------------------------------------------------------------------------

_NA_VALUES = frozenset({"n/a", "[n/a]", "not supported", "none", "", "-"})


def _parse_float(raw: str) -> Optional[float]:
    """Convert a raw nvidia-smi CSV field to float, returning None on N/A."""
    cleaned = raw.strip().lower()
    if cleaned in _NA_VALUES:
        return None
    # nvidia-smi sometimes returns values like "123.45 W" — strip trailing units
    # that aren't already stripped by --nounits. Take only the first token.
    first_token = cleaned.split()[0] if cleaned.split() else cleaned
    try:
        return float(first_token)
    except ValueError:
        logger.debug("Cannot parse float from %r", raw)
        return None


def _parse_mb_to_bytes(raw: str) -> Optional[float]:
    """Convert MiB string to bytes (nvidia-smi reports memory in MiB)."""
    val = _parse_float(raw)
    if val is None:
        return None
    return val * 1024 * 1024


def _parse_int(raw: str) -> Optional[int]:
    val = _parse_float(raw)
    if val is None:
        return None
    return int(val)


# ---------------------------------------------------------------------------
# Username resolution from PID
# ---------------------------------------------------------------------------

def _uid_from_proc(pid: str) -> Optional[int]:
    """Read the real UID of a process from /proc/<pid>/status."""
    try:
        status_path = Path(f"/proc/{pid}/status")
        with status_path.open() as fh:
            for line in fh:
                if line.startswith("Uid:"):
                    # Uid: real  effective  saved  filesystem
                    parts = line.split()
                    return int(parts[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _username_from_uid(uid: int) -> str:
    """Resolve a UID to a username, falling back to the numeric string."""
    try:
        return pwd.getpwuid(uid).pw_name
    except (KeyError, TypeError):
        return str(uid)


def _get_process_username(pid: str) -> str:
    """Return the username that owns *pid*, or 'unknown'."""
    uid = _uid_from_proc(pid)
    if uid is None:
        return "unknown"
    return _username_from_uid(uid)


# ---------------------------------------------------------------------------
# Prometheus metric definitions
# ---------------------------------------------------------------------------

# Availability
SMI_AVAILABLE = Gauge(
    "nvidia_smi_available",
    "1 if nvidia-smi is available and responsive, 0 otherwise",
)

# Driver / CUDA info (value=1, metadata via labels)
DRIVER_INFO = Gauge(
    "nvidia_driver_info",
    "NVIDIA driver and CUDA version info",
    ["driver_version", "cuda_version"],
)

# Per-GPU gauges — all carry gpu_index + gpu_name labels
_GPU_LABELS = ["gpu_index", "gpu_name", "gpu_uuid"]

GPU_TEMP = Gauge(
    "nvidia_gpu_temperature_celsius",
    "GPU core temperature in Celsius",
    _GPU_LABELS,
)
GPU_UTIL = Gauge(
    "nvidia_gpu_utilization_percent",
    "GPU core utilization in percent",
    _GPU_LABELS,
)
GPU_MEM_UTIL = Gauge(
    "nvidia_gpu_memory_utilization_percent",
    "GPU memory bus utilization in percent",
    _GPU_LABELS,
)
GPU_MEM_USED = Gauge(
    "nvidia_gpu_memory_used_bytes",
    "GPU used VRAM in bytes",
    _GPU_LABELS,
)
GPU_MEM_FREE = Gauge(
    "nvidia_gpu_memory_free_bytes",
    "GPU free VRAM in bytes",
    _GPU_LABELS,
)
GPU_MEM_TOTAL = Gauge(
    "nvidia_gpu_memory_total_bytes",
    "GPU total VRAM in bytes",
    _GPU_LABELS,
)
GPU_POWER_DRAW = Gauge(
    "nvidia_gpu_power_draw_watts",
    "GPU current power draw in Watts",
    _GPU_LABELS,
)
GPU_POWER_LIMIT = Gauge(
    "nvidia_gpu_power_limit_watts",
    "GPU power limit in Watts",
    _GPU_LABELS,
)
GPU_FAN_SPEED = Gauge(
    "nvidia_gpu_fan_speed_percent",
    "GPU fan speed in percent",
    _GPU_LABELS,
)
GPU_CLOCK_GRAPHICS = Gauge(
    "nvidia_gpu_clock_graphics_mhz",
    "GPU graphics clock in MHz",
    _GPU_LABELS,
)
GPU_CLOCK_SM = Gauge(
    "nvidia_gpu_clock_sm_mhz",
    "GPU SM (shader) clock in MHz",
    _GPU_LABELS,
)
GPU_CLOCK_MEM = Gauge(
    "nvidia_gpu_clock_memory_mhz",
    "GPU memory clock in MHz",
    _GPU_LABELS,
)
GPU_CLOCK_VIDEO = Gauge(
    "nvidia_gpu_clock_video_mhz",
    "GPU video encoder clock in MHz",
    _GPU_LABELS,
)
GPU_ECC_SINGLE = Gauge(
    "nvidia_gpu_ecc_errors_single_bit_total",
    "Total single-bit ECC errors (volatile)",
    _GPU_LABELS,
)
GPU_ECC_DOUBLE = Gauge(
    "nvidia_gpu_ecc_errors_double_bit_total",
    "Total double-bit ECC errors (volatile)",
    _GPU_LABELS,
)
GPU_PCIE_GEN = Gauge(
    "nvidia_gpu_pcie_link_gen",
    "Current PCIe link generation",
    _GPU_LABELS,
)
GPU_PCIE_WIDTH = Gauge(
    "nvidia_gpu_pcie_link_width",
    "Current PCIe link width (number of lanes)",
    _GPU_LABELS,
)
GPU_ENC_UTIL = Gauge(
    "nvidia_gpu_encoder_utilization_percent",
    "GPU hardware encoder utilization in percent",
    _GPU_LABELS,
)
GPU_DEC_UTIL = Gauge(
    "nvidia_gpu_decoder_utilization_percent",
    "GPU hardware decoder utilization in percent",
    _GPU_LABELS,
)
GPU_COMPUTE_MODE = Gauge(
    "nvidia_gpu_compute_mode",
    "GPU compute mode (0=Default, 1=Exclusive Thread, 2=Prohibited, 3=Exclusive Process)",
    _GPU_LABELS + ["compute_mode_str"],
)
GPU_MEM_RESERVED = Gauge(
    "nvidia_gpu_memory_reserved_bytes",
    "GPU memory reserved by driver in bytes",
    _GPU_LABELS,
)
GPU_POWER_MAX = Gauge(
    "nvidia_gpu_power_max_limit_watts",
    "GPU maximum allowed power limit in Watts",
    _GPU_LABELS,
)
GPU_POWER_MIN = Gauge(
    "nvidia_gpu_power_min_limit_watts",
    "GPU minimum allowed power limit in Watts",
    _GPU_LABELS,
)
GPU_POWER_ENFORCED = Gauge(
    "nvidia_gpu_power_enforced_limit_watts",
    "GPU currently enforced power limit in Watts",
    _GPU_LABELS,
)
GPU_CLOCK_MAX_GRAPHICS = Gauge(
    "nvidia_gpu_clock_max_graphics_mhz",
    "Maximum GPU graphics clock in MHz",
    _GPU_LABELS,
)
GPU_CLOCK_MAX_SM = Gauge(
    "nvidia_gpu_clock_max_sm_mhz",
    "Maximum GPU SM clock in MHz",
    _GPU_LABELS,
)
GPU_CLOCK_MAX_MEM = Gauge(
    "nvidia_gpu_clock_max_memory_mhz",
    "Maximum GPU memory clock in MHz",
    _GPU_LABELS,
)
GPU_PSTATE = Gauge(
    "nvidia_gpu_pstate",
    "GPU performance state (0=P0 max performance, 12=P12 min performance)",
    _GPU_LABELS,
)
GPU_THROTTLE_SW_POWER = Gauge(
    "nvidia_gpu_throttle_sw_power_cap",
    "1 if GPU is throttled due to software power cap, 0 otherwise",
    _GPU_LABELS,
)
GPU_THROTTLE_HW_SLOWDOWN = Gauge(
    "nvidia_gpu_throttle_hw_slowdown",
    "1 if GPU is in HW slowdown (thermal or power), 0 otherwise",
    _GPU_LABELS,
)
GPU_THROTTLE_HW_THERMAL = Gauge(
    "nvidia_gpu_throttle_hw_thermal_slowdown",
    "1 if GPU is throttled due to HW thermal limit, 0 otherwise",
    _GPU_LABELS,
)
GPU_THROTTLE_SW_THERMAL = Gauge(
    "nvidia_gpu_throttle_sw_thermal_slowdown",
    "1 if GPU is throttled due to SW thermal limit, 0 otherwise",
    _GPU_LABELS,
)
# Static GPU info (labels only, value=1)
GPU_STATIC_INFO = Gauge(
    "nvidia_gpu_static_info",
    "Static GPU properties: vbios version, serial number, compute capability",
    _GPU_LABELS + ["vbios_version", "serial", "compute_cap"],
)

# Per-process memory
_PROC_LABELS = ["gpu_index", "gpu_name", "pid", "process_name", "username", "type"]
GPU_PROC_MEM = Gauge(
    "nvidia_gpu_process_memory_used_bytes",
    "Memory used by a process on the GPU in bytes",
    _PROC_LABELS,
)

# GPU count
GPU_COUNT = Gauge(
    "nvidia_gpu_count",
    "Total number of GPUs detected by nvidia-smi",
)

# Exporter health metrics
SCRAPE_DURATION = Gauge(
    "nvidia_exporter_scrape_duration_seconds",
    "Duration of the last nvidia-smi metric collection in seconds",
)
LAST_SCRAPE_TS = Gauge(
    "nvidia_exporter_last_scrape_timestamp_seconds",
    "Unix timestamp of the last successful nvidia-smi scrape",
)

# ── Datacenter / High-end GPU metrics ────────────────────────────────────────

# ECC Aggregate (persistent across driver restart, unlike volatile)
GPU_ECC_AGG_SINGLE = Gauge(
    "nvidia_gpu_ecc_errors_aggregate_single_bit_total",
    "Aggregate single-bit ECC errors (persists across driver restart)",
    _GPU_LABELS,
)
GPU_ECC_AGG_DOUBLE = Gauge(
    "nvidia_gpu_ecc_errors_aggregate_double_bit_total",
    "Aggregate double-bit ECC errors (persists across driver restart)",
    _GPU_LABELS,
)

# Retired pages — memory pages retired due to ECC errors (A100/H100/datacenter)
GPU_RETIRED_SBE = Gauge(
    "nvidia_gpu_retired_pages_single_bit_ecc_total",
    "Number of memory pages retired due to single-bit ECC errors",
    _GPU_LABELS,
)
GPU_RETIRED_DBE = Gauge(
    "nvidia_gpu_retired_pages_double_bit_ecc_total",
    "Number of memory pages retired due to double-bit ECC errors",
    _GPU_LABELS,
)
GPU_RETIRED_PENDING = Gauge(
    "nvidia_gpu_retired_pages_pending",
    "1 if page retirement requires reboot, 0 otherwise",
    _GPU_LABELS,
)

# BAR1 memory — used for direct GPU memory mapping (important for datacenter NVLink/peer access)
GPU_BAR1_USED = Gauge(
    "nvidia_gpu_bar1_memory_used_bytes",
    "BAR1 memory used (direct GPU memory mapping) in bytes",
    _GPU_LABELS,
)
GPU_BAR1_FREE = Gauge(
    "nvidia_gpu_bar1_memory_free_bytes",
    "BAR1 memory free in bytes",
    _GPU_LABELS,
)
GPU_BAR1_TOTAL = Gauge(
    "nvidia_gpu_bar1_memory_total_bytes",
    "BAR1 memory total in bytes",
    _GPU_LABELS,
)

# Remapped rows (HBM memory repair on A100/H100)
GPU_REMAPPED_ROWS_CE = Gauge(
    "nvidia_gpu_remapped_rows_correctable_total",
    "Number of rows remapped due to correctable errors (A100/H100 HBM)",
    _GPU_LABELS,
)
GPU_REMAPPED_ROWS_UE = Gauge(
    "nvidia_gpu_remapped_rows_uncorrectable_total",
    "Number of rows remapped due to uncorrectable errors (A100/H100 HBM)",
    _GPU_LABELS,
)
GPU_REMAPPED_ROWS_PENDING = Gauge(
    "nvidia_gpu_remapped_rows_pending",
    "1 if row remap requires reboot, 0 otherwise",
    _GPU_LABELS,
)
GPU_REMAPPED_ROWS_FAILURE = Gauge(
    "nvidia_gpu_remapped_rows_failure",
    "1 if row remapping has failed, 0 otherwise",
    _GPU_LABELS,
)

# NVLink metrics — per link bandwidth and errors
# NVLink labels include link_id
_NVLINK_LABELS = _GPU_LABELS + ["link_id"]
GPU_NVLINK_STATE = Gauge(
    "nvidia_gpu_nvlink_state",
    "NVLink link state (1=active, 0=inactive)",
    _NVLINK_LABELS,
)
GPU_NVLINK_TX_BYTES = Gauge(
    "nvidia_gpu_nvlink_tx_bytes_total",
    "NVLink TX throughput in bytes (cumulative counter)",
    _NVLINK_LABELS,
)
GPU_NVLINK_RX_BYTES = Gauge(
    "nvidia_gpu_nvlink_rx_bytes_total",
    "NVLink RX throughput in bytes (cumulative counter)",
    _NVLINK_LABELS,
)
GPU_NVLINK_REPLAY_ERRORS = Gauge(
    "nvidia_gpu_nvlink_replay_errors_total",
    "NVLink replay error count per link",
    _NVLINK_LABELS,
)
GPU_NVLINK_RECOVERY_ERRORS = Gauge(
    "nvidia_gpu_nvlink_recovery_errors_total",
    "NVLink recovery error count per link",
    _NVLINK_LABELS,
)
GPU_NVLINK_CRC_ERRORS = Gauge(
    "nvidia_gpu_nvlink_crc_errors_total",
    "NVLink CRC error count per link",
    _NVLINK_LABELS,
)

# NVLink summary
GPU_NVLINK_COUNT = Gauge(
    "nvidia_gpu_nvlink_link_count",
    "Number of active NVLink links on this GPU",
    _GPU_LABELS,
)

# MIG (Multi-Instance GPU) — A100/H100/A30
GPU_MIG_MODE = Gauge(
    "nvidia_gpu_mig_mode_enabled",
    "1 if MIG mode is enabled on this GPU, 0 otherwise",
    _GPU_LABELS,
)
GPU_MIG_INSTANCE_COUNT = Gauge(
    "nvidia_gpu_mig_instance_count",
    "Number of active MIG GPU instances",
    _GPU_LABELS,
)

# Per-MIG-instance metrics (if MIG enabled)
_MIG_LABELS = _GPU_LABELS + ["mig_gi", "mig_ci", "mig_profile"]
GPU_MIG_MEM_USED = Gauge(
    "nvidia_gpu_mig_memory_used_bytes",
    "MIG instance used memory in bytes",
    _MIG_LABELS,
)
GPU_MIG_MEM_TOTAL = Gauge(
    "nvidia_gpu_mig_memory_total_bytes",
    "MIG instance total memory in bytes",
    _MIG_LABELS,
)
GPU_MIG_UTIL = Gauge(
    "nvidia_gpu_mig_utilization_percent",
    "MIG instance compute utilization percent",
    _MIG_LABELS,
)

# Datacenter-specific additional metrics
GPU_TEMP_MEM = Gauge(
    "nvidia_gpu_memory_temperature_celsius",
    "GPU memory (HBM) temperature in Celsius (datacenter GPUs)",
    _GPU_LABELS,
)
GPU_TEMP_GPU_TARGET = Gauge(
    "nvidia_gpu_slowdown_temp_celsius",
    "GPU slowdown temperature threshold in Celsius",
    _GPU_LABELS,
)
GPU_TEMP_SHUTDOWN = Gauge(
    "nvidia_gpu_shutdown_temp_celsius",
    "GPU shutdown temperature threshold in Celsius",
    _GPU_LABELS,
)

# Power total energy consumption (Joules, counter — datacenter tracking)
GPU_TOTAL_ENERGY = Gauge(
    "nvidia_gpu_total_energy_consumption_joules",
    "Total energy consumed by GPU since last driver load in Joules",
    _GPU_LABELS,
)

# ---------------------------------------------------------------------------
# Compute mode string map
# ---------------------------------------------------------------------------

_COMPUTE_MODE_MAP = {
    "default": 0,
    "exclusive_thread": 1,
    "prohibited": 2,
    "exclusive_process": 3,
}


def _compute_mode_int(raw: str) -> int:
    cleaned = raw.strip().lower().replace(" ", "_")
    return _COMPUTE_MODE_MAP.get(cleaned, 0)


# ---------------------------------------------------------------------------
# nvidia-smi query fields and result mapping
# ---------------------------------------------------------------------------

# Each tuple: (query_field, parse_function, target_gauge_or_None)
# Gauges that need special treatment (pcie kb/s → bytes/s conversion) are
# handled explicitly in the collection loop.
_GPU_QUERY_FIELDS = [
    "index",                                    # 0
    "name",                                     # 1
    "uuid",                                     # 2
    "driver_version",                           # 3
    "temperature.gpu",                          # 4
    "utilization.gpu",                          # 5
    "utilization.memory",                       # 6
    "memory.used",                              # 7  MiB → bytes
    "memory.free",                              # 8  MiB → bytes
    "memory.total",                             # 9  MiB → bytes
    "memory.reserved",                          # 10 MiB → bytes
    "power.draw",                               # 11
    "power.limit",                              # 12
    "power.max_limit",                          # 13
    "power.min_limit",                          # 14
    "enforced.power.limit",                     # 15
    "fan.speed",                                # 16
    "clocks.current.graphics",                  # 17
    "clocks.current.sm",                        # 18
    "clocks.current.memory",                    # 19
    "clocks.current.video",                     # 20
    "clocks.max.graphics",                      # 21
    "clocks.max.sm",                            # 22
    "clocks.max.memory",                        # 23
    "ecc.errors.corrected.volatile.total",      # 24 single-bit
    "ecc.errors.uncorrected.volatile.total",    # 25 double-bit
    "pcie.link.gen.current",                    # 26
    "pcie.link.width.current",                  # 27
    "utilization.encoder",                      # 28
    "utilization.decoder",                      # 29
    "compute_mode",                             # 30
    "pstate",                                   # 31 e.g. "P2"
    "clocks_throttle_reasons.sw_power_cap",     # 32 "Active"/"Not Active"
    "clocks_throttle_reasons.hw_slowdown",      # 33
    "clocks_throttle_reasons.hw_thermal_slowdown",  # 34
    "clocks_throttle_reasons.sw_thermal_slowdown",  # 35
    "vbios_version",                            # 36 (label)
    "serial",                                   # 37 (label)
    "compute_cap",                              # 38 (label)
]

# Datacenter-only fields — queried separately so failure on consumer GPUs
# doesn't break the base query. Each tuple: (field, index_offset_from_0)
_DC_QUERY_FIELDS = [
    "ecc.errors.corrected.aggregate.total",     # 0  aggregate single-bit
    "ecc.errors.uncorrected.aggregate.total",   # 1  aggregate double-bit
    "retired_pages.single_bit_ecc.count",       # 2  retired pages SBE
    "retired_pages.double_bit_ecc.count",       # 3  retired pages DBE
    "retired_pages.pending",                    # 4  Yes/No
    "bar1_memory.used",                         # 5  MiB
    "bar1_memory.free",                         # 6  MiB
    "bar1_memory.total",                        # 7  MiB
    "remapped_rows.correctable",                # 8
    "remapped_rows.uncorrectable",              # 9
    "remapped_rows.pending",                    # 10 Yes/No
    "remapped_rows.failure",                    # 11 Yes/No
    "temperature.memory",                       # 12 HBM temp
    "temperature.gpu.slowdown",                 # 13 slowdown threshold
    "temperature.gpu.shutdown",                 # 14 shutdown threshold
    "power.total_energy_consumption",           # 15 mJ → J
]

# Probe once at startup to see if datacenter fields are supported
_dc_fields_supported: Optional[bool] = None


def _check_dc_fields() -> bool:
    """Return True if this driver supports datacenter-specific query fields."""
    global _dc_fields_supported
    if _dc_fields_supported is not None:
        return _dc_fields_supported
    test_out = _run_cmd([
        "nvidia-smi",
        f"--query-gpu=index,{','.join(_DC_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ])
    _dc_fields_supported = test_out is not None
    if _dc_fields_supported:
        logger.info("Datacenter GPU fields supported — extended metrics enabled.")
    else:
        logger.info("Datacenter GPU fields not supported on this hardware — using base metrics only.")
    return _dc_fields_supported


def _build_query_string() -> str:
    return ",".join(_GPU_QUERY_FIELDS)


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------

class NvidiaCollector:
    """
    Collects all GPU metrics from nvidia-smi and updates Prometheus gauges.
    Designed to be called on a background thread at SCRAPE_INTERVAL.
    """

    # Track which label-sets we have previously set for process metrics so we
    # can clear stale entries (processes that have exited).
    _proc_label_sets: Set[Tuple] = set()
    _driver_label_sets: Set[Tuple] = set()

    def collect(self) -> None:
        """Entry point — called once per scrape cycle."""
        if not _nvidia_smi_available():
            SMI_AVAILABLE.set(0)
            logger.warning(
                "nvidia-smi is not available or returned an error. "
                "Only nvidia_smi_available=0 will be exposed."
            )
            return

        SMI_AVAILABLE.set(1)

        try:
            self._collect_gpu_metrics()
        except Exception as exc:
            logger.error("Unhandled error in GPU metric collection: %s", exc, exc_info=True)

        try:
            self._collect_process_metrics()
        except Exception as exc:
            logger.error("Unhandled error in process metric collection: %s", exc, exc_info=True)

        try:
            self._collect_nvlink_metrics()
        except Exception as exc:
            logger.error("Unhandled error in NVLink metric collection: %s", exc, exc_info=True)

        try:
            self._collect_mig_metrics()
        except Exception as exc:
            logger.error("Unhandled error in MIG metric collection: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # GPU-level metrics
    # ------------------------------------------------------------------

    def _collect_gpu_metrics(self) -> None:
        query = _build_query_string()
        output = _run_cmd(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        )
        if output is None:
            logger.warning("nvidia-smi GPU query returned no output.")
            GPU_COUNT.set(0)
            return

        # Collect datacenter fields separately (may not be supported on all GPUs)
        dc_supported = _check_dc_fields()
        dc_output: Optional[str] = None
        if dc_supported:
            dc_query = f"index,{','.join(_DC_QUERY_FIELDS)}"
            dc_output = _run_cmd(
                ["nvidia-smi", f"--query-gpu={dc_query}", "--format=csv,noheader,nounits"]
            )
        # Build dc_parts_map: gpu_index → list of dc field values
        dc_parts_map: Dict[str, List[str]] = {}
        if dc_output:
            for dc_line in dc_output.strip().splitlines():
                dc_parts = [p.strip() for p in dc_line.split(",")]
                if dc_parts:
                    dc_parts_map[dc_parts[0]] = dc_parts[1:]

        seen_driver_labels: Set[Tuple] = set()
        gpu_count = 0

        for line in output.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]

            # Ensure we have at least enough fields; pad with "N/A" if short
            while len(parts) < len(_GPU_QUERY_FIELDS):
                parts.append("N/A")

            try:
                gpu_idx_raw = parts[0].strip()
                dc_parts = dc_parts_map.get(gpu_idx_raw, [])
                self._process_gpu_line(parts, seen_driver_labels, dc_parts)
                gpu_count += 1
            except Exception as exc:
                logger.debug("Error processing GPU line %r: %s", line, exc)

        GPU_COUNT.set(gpu_count)

        # Clear stale driver info labels
        for old_labels in self._driver_label_sets - seen_driver_labels:
            try:
                DRIVER_INFO.remove(*old_labels)
            except Exception:
                pass
        self._driver_label_sets = seen_driver_labels

    def _process_gpu_line(
        self, parts: List[str], seen_driver_labels: Set[Tuple],
        dc_parts: Optional[List[str]] = None,
    ) -> None:
        # --- Identity fields ---
        gpu_index = parts[0].strip()
        gpu_name  = parts[1].strip()
        gpu_uuid  = parts[2].strip()

        driver_version = parts[3].strip()

        # Get CUDA version once from nvidia-smi full output (cached)
        cuda_version = _get_cuda_version()

        # Driver info gauge (set once per unique driver/cuda combo)
        driver_label_key = (driver_version, cuda_version)
        if driver_label_key not in seen_driver_labels:
            DRIVER_INFO.labels(
                driver_version=driver_version,
                cuda_version=cuda_version,
            ).set(1)
            seen_driver_labels.add(driver_label_key)

        lbl = dict(gpu_index=gpu_index, gpu_name=gpu_name, gpu_uuid=gpu_uuid)

        # --- Helper closures ---
        def _set_gauge(gauge: Gauge, raw: str, converter=_parse_float) -> None:
            val = converter(raw)
            if val is not None:
                gauge.labels(**lbl).set(val)

        def _set_gauge_or_zero(gauge: Gauge, raw: str) -> None:
            """Set gauge, defaulting to 0 for N/A (e.g. ECC on non-ECC GPUs)."""
            val = _parse_float(raw)
            gauge.labels(**lbl).set(val if val is not None else 0.0)

        def _set_mb_gauge(gauge: Gauge, raw: str) -> None:
            val = _parse_mb_to_bytes(raw)
            if val is not None:
                gauge.labels(**lbl).set(val)

        # --- Temperature ---
        _set_gauge(GPU_TEMP, parts[4])

        # --- Utilization ---
        _set_gauge(GPU_UTIL, parts[5])
        _set_gauge(GPU_MEM_UTIL, parts[6])

        # --- Memory (MiB → bytes) ---
        _set_mb_gauge(GPU_MEM_USED, parts[7])
        _set_mb_gauge(GPU_MEM_FREE, parts[8])
        _set_mb_gauge(GPU_MEM_TOTAL, parts[9])
        _set_mb_gauge(GPU_MEM_RESERVED, parts[10])

        # --- Power ---
        _set_gauge(GPU_POWER_DRAW, parts[11])
        _set_gauge(GPU_POWER_LIMIT, parts[12])
        _set_gauge(GPU_POWER_MAX, parts[13])
        _set_gauge(GPU_POWER_MIN, parts[14])
        _set_gauge(GPU_POWER_ENFORCED, parts[15])

        # --- Fan speed ---
        _set_gauge(GPU_FAN_SPEED, parts[16])

        # --- Clocks current ---
        _set_gauge(GPU_CLOCK_GRAPHICS, parts[17])
        _set_gauge(GPU_CLOCK_SM, parts[18])
        _set_gauge(GPU_CLOCK_MEM, parts[19])
        _set_gauge(GPU_CLOCK_VIDEO, parts[20])

        # --- Clocks max ---
        _set_gauge(GPU_CLOCK_MAX_GRAPHICS, parts[21])
        _set_gauge(GPU_CLOCK_MAX_SM, parts[22])
        _set_gauge(GPU_CLOCK_MAX_MEM, parts[23])

        # --- ECC errors (default 0 for non-ECC GPUs that return [N/A]) ---
        _set_gauge_or_zero(GPU_ECC_SINGLE, parts[24])
        _set_gauge_or_zero(GPU_ECC_DOUBLE, parts[25])

        # --- PCIe ---
        _set_gauge(GPU_PCIE_GEN, parts[26])
        _set_gauge(GPU_PCIE_WIDTH, parts[27])

        # --- Encoder / Decoder utilization ---
        _set_gauge(GPU_ENC_UTIL, parts[28])
        _set_gauge(GPU_DEC_UTIL, parts[29])

        # --- Compute mode ---
        compute_mode_raw = parts[30].strip()
        compute_mode_int_val = _compute_mode_int(compute_mode_raw)
        compute_mode_lbl = dict(**lbl, compute_mode_str=compute_mode_raw)
        try:
            GPU_COMPUTE_MODE.labels(**compute_mode_lbl).set(compute_mode_int_val)
        except Exception as exc:
            logger.debug("Could not set compute_mode gauge: %s", exc)

        # --- Performance state (P0..P12 → int) ---
        pstate_raw = parts[31].strip()
        try:
            GPU_PSTATE.labels(**lbl).set(int(pstate_raw.lstrip("P")))
        except (ValueError, IndexError):
            GPU_PSTATE.labels(**lbl).set(0)

        # --- Throttle reasons (Active=1, else=0) ---
        def _active_to_int(raw: str) -> int:
            s = raw.strip().lower()
            return 1 if "active" in s and "not" not in s else 0

        GPU_THROTTLE_SW_POWER.labels(**lbl).set(_active_to_int(parts[32]))
        GPU_THROTTLE_HW_SLOWDOWN.labels(**lbl).set(_active_to_int(parts[33]))
        GPU_THROTTLE_HW_THERMAL.labels(**lbl).set(_active_to_int(parts[34]))
        GPU_THROTTLE_SW_THERMAL.labels(**lbl).set(_active_to_int(parts[35]))

        # --- Static info (vbios, serial, compute_cap) — label-only gauge ---
        vbios = parts[36].strip()
        serial = parts[37].strip()
        compute_cap = parts[38].strip()
        try:
            GPU_STATIC_INFO.labels(
                **lbl, vbios_version=vbios, serial=serial, compute_cap=compute_cap
            ).set(1)
        except Exception as exc:
            logger.debug("Could not set static info gauge: %s", exc)

        # --- Datacenter-only fields (only if supported by this driver/GPU) ---
        if dc_parts:
            def _yes_to_int(raw: str) -> int:
                return 1 if raw.strip().lower() in ("yes", "1", "true") else 0

            def _dc_set(gauge, idx, converter=_set_gauge_or_zero):
                if idx < len(dc_parts):
                    converter(gauge, dc_parts[idx])

            def _dc_mb(gauge, idx):
                if idx < len(dc_parts):
                    _set_mb_gauge(gauge, dc_parts[idx])

            def _dc_bool(gauge, idx):
                if idx < len(dc_parts):
                    gauge.labels(**lbl).set(_yes_to_int(dc_parts[idx]))

            # Aggregate ECC
            _dc_set(GPU_ECC_AGG_SINGLE, 0)
            _dc_set(GPU_ECC_AGG_DOUBLE, 1)
            # Retired pages
            _dc_set(GPU_RETIRED_SBE, 2)
            _dc_set(GPU_RETIRED_DBE, 3)
            _dc_bool(GPU_RETIRED_PENDING, 4)
            # BAR1 memory
            _dc_mb(GPU_BAR1_USED, 5)
            _dc_mb(GPU_BAR1_FREE, 6)
            _dc_mb(GPU_BAR1_TOTAL, 7)
            # Remapped rows (HBM)
            _dc_set(GPU_REMAPPED_ROWS_CE, 8)
            _dc_set(GPU_REMAPPED_ROWS_UE, 9)
            _dc_bool(GPU_REMAPPED_ROWS_PENDING, 10)
            _dc_bool(GPU_REMAPPED_ROWS_FAILURE, 11)
            # Datacenter temperatures
            if 12 < len(dc_parts): _set_gauge(GPU_TEMP_MEM, dc_parts[12])
            if 13 < len(dc_parts): _set_gauge(GPU_TEMP_GPU_TARGET, dc_parts[13])
            if 14 < len(dc_parts): _set_gauge(GPU_TEMP_SHUTDOWN, dc_parts[14])
            # Total energy (mJ → J)
            if 15 < len(dc_parts):
                energy_mj = _parse_float(dc_parts[15])
                if energy_mj is not None:
                    GPU_TOTAL_ENERGY.labels(**lbl).set(energy_mj / 1000.0)

    # ------------------------------------------------------------------
    # Per-process metrics
    # ------------------------------------------------------------------

    def _collect_process_metrics(self) -> None:
        """
        Collect per-process GPU memory usage.

        We use two nvidia-smi sub-commands and merge the results:
          1. --query-compute-apps  → compute (CUDA) processes
          2. --query-accounted-apps (fallback: pmon)  → additional detail

        For each process we look up the owning username via /proc.
        """
        new_proc_label_sets: Set[Tuple] = set()

        # Fetch GPU name index map so we can add gpu_name to process labels
        gpu_name_map = self._fetch_gpu_name_map()

        # --- Compute processes ---
        compute_out = _run_cmd(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory,name",
                "--format=csv,noheader,nounits",
            ]
        )
        if compute_out:
            for line in compute_out.strip().splitlines():
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                try:
                    gpu_uuid_proc = parts[0]
                    pid           = parts[1]
                    used_mem_mib  = parts[2]
                    proc_name     = parts[3]

                    gpu_index_proc = gpu_name_map.get(gpu_uuid_proc, {}).get("index", "0")
                    gpu_name_proc  = gpu_name_map.get(gpu_uuid_proc, {}).get("name", "unknown")

                    mem_bytes = _parse_mb_to_bytes(used_mem_mib)
                    if mem_bytes is None:
                        mem_bytes = 0.0

                    username = _get_process_username(pid)

                    label_tuple = (gpu_index_proc, gpu_name_proc, pid, proc_name, username, "C")
                    GPU_PROC_MEM.labels(
                        gpu_index=gpu_index_proc,
                        gpu_name=gpu_name_proc,
                        pid=pid,
                        process_name=proc_name,
                        username=username,
                        type="C",
                    ).set(mem_bytes)
                    new_proc_label_sets.add(label_tuple)

                except Exception as exc:
                    logger.debug("Error processing compute app line %r: %s", line, exc)

        # --- Graphics / display processes via pmon ---
        # pmon output format: gpu  pid  type  sm  mem  enc  dec  command
        pmon_out = _run_cmd(["nvidia-smi", "pmon", "-c", "1", "-s", "mu"])
        if pmon_out:
            for line in pmon_out.strip().splitlines():
                # Skip header lines (start with '#') or empty
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                # Expected columns: gpu  pid  type  fb  ccpm  sm  mem  enc  dec  command
                # At minimum we need: gpu(0) pid(1) type(2) fb(3) command(-1)
                if len(parts) < 5:
                    continue
                try:
                    gpu_idx_pmon  = parts[0]
                    pid_pmon      = parts[1]
                    proc_type     = parts[2]   # C, G, or C+G
                    fb_mib        = parts[3]   # frame buffer usage in MiB

                    # Process command is the last field
                    proc_name_pmon = parts[-1]

                    # Skip if pid is '-' (idle slot)
                    if pid_pmon in ("-", "N/A"):
                        continue

                    # Resolve gpu_name
                    gpu_name_pmon = self._gpu_name_from_index(gpu_name_map, gpu_idx_pmon)

                    # Determine process type label
                    # pmon uses: C=compute, G=graphics, C+G=both
                    if proc_type in ("C", "C+G"):
                        ptype = "C"
                    elif proc_type == "G":
                        ptype = "G"
                    else:
                        ptype = proc_type[:1] if proc_type else "?"

                    mem_bytes = _parse_mb_to_bytes(fb_mib)
                    if mem_bytes is None:
                        mem_bytes = 0.0

                    username_pmon = _get_process_username(pid_pmon)

                    label_tuple = (gpu_idx_pmon, gpu_name_pmon, pid_pmon, proc_name_pmon, username_pmon, ptype)

                    # Only set if not already seen from compute-apps query (prefer that)
                    if label_tuple not in new_proc_label_sets:
                        GPU_PROC_MEM.labels(
                            gpu_index=gpu_idx_pmon,
                            gpu_name=gpu_name_pmon,
                            pid=pid_pmon,
                            process_name=proc_name_pmon,
                            username=username_pmon,
                            type=ptype,
                        ).set(mem_bytes)
                        new_proc_label_sets.add(label_tuple)

                except Exception as exc:
                    logger.debug("Error processing pmon line %r: %s", line, exc)

        # Remove stale process entries (processes that have exited)
        for old_labels in self._proc_label_sets - new_proc_label_sets:
            try:
                GPU_PROC_MEM.remove(*old_labels)
            except Exception:
                pass

        self._proc_label_sets = new_proc_label_sets

    def _collect_nvlink_metrics(self) -> None:
        """
        Collect NVLink metrics per GPU per link.
        Uses: nvidia-smi nvlink --status and --errorcounters
        Only runs if NVLink-capable GPUs are detected.
        """
        import re as _re

        # Check if nvlink subcommand works at all
        status_out = _run_cmd(["nvidia-smi", "nvlink", "--status", "--id", "0"], timeout=10)
        if status_out is None:
            return  # No NVLink on this system — silently skip

        # Fetch GPU index map for label resolution
        gpu_map = self._fetch_gpu_name_map()
        # Build index → (name, uuid) map
        idx_map: Dict[str, Tuple] = {}
        for uuid, info in gpu_map.items():
            idx_map[info["index"]] = (info["name"], uuid)

        # Run nvlink status for each GPU index
        for gpu_idx, (gpu_name, gpu_uuid) in idx_map.items():
            lbl_base = dict(gpu_index=gpu_idx, gpu_name=gpu_name, gpu_uuid=gpu_uuid)
            active_links = 0

            # Status
            s_out = _run_cmd(["nvidia-smi", "nvlink", "--status", "--id", gpu_idx], timeout=10)
            if s_out:
                for line in s_out.splitlines():
                    # Lines like: "Link 0: 25.000 GB/s" or "Link 0: <inactive>"
                    m = _re.match(r"\s*Link\s+(\d+):\s*(.*)", line, _re.IGNORECASE)
                    if m:
                        link_id = m.group(1)
                        state_str = m.group(2).strip().lower()
                        state_val = 0 if "inactive" in state_str or "n/a" in state_str else 1
                        lbl_link = dict(**lbl_base, link_id=link_id)
                        try:
                            GPU_NVLINK_STATE.labels(**lbl_link).set(state_val)
                            if state_val:
                                active_links += 1
                        except Exception:
                            pass

            GPU_NVLINK_COUNT.labels(**lbl_base).set(active_links)

            # Error counters
            e_out = _run_cmd(
                ["nvidia-smi", "nvlink", "--errorcounters", "--id", gpu_idx], timeout=10
            )
            if e_out:
                for line in e_out.splitlines():
                    # Lines like: "Link 0: Replay Errors: 0"
                    m_link = _re.match(r"\s*Link\s+(\d+):(.+)", line, _re.IGNORECASE)
                    if not m_link:
                        continue
                    link_id = m_link.group(1)
                    rest = m_link.group(2).lower()
                    lbl_link = dict(**lbl_base, link_id=link_id)
                    val_m = _re.search(r"(\d+)\s*$", rest)
                    val = int(val_m.group(1)) if val_m else 0
                    try:
                        if "replay" in rest:
                            GPU_NVLINK_REPLAY_ERRORS.labels(**lbl_link).set(val)
                        elif "recovery" in rest:
                            GPU_NVLINK_RECOVERY_ERRORS.labels(**lbl_link).set(val)
                        elif "crc" in rest:
                            GPU_NVLINK_CRC_ERRORS.labels(**lbl_link).set(val)
                    except Exception:
                        pass

            # Bandwidth counters via nvlink --setcontrol bandwidth if available
            # Use --query-gpu=nvlink.bandwidth.c0.tx if nvidia-smi supports it
            for link_id_int in range(active_links):
                tx_field = f"nvlink.bandwidth.c{link_id_int}.tx"
                rx_field = f"nvlink.bandwidth.c{link_id_int}.rx"
                bw_out = _run_cmd([
                    "nvidia-smi",
                    f"--query-gpu=index,{tx_field},{rx_field}",
                    "--format=csv,noheader,nounits",
                    "--id", gpu_idx,
                ], timeout=10)
                if bw_out:
                    for bw_line in bw_out.strip().splitlines():
                        bw_parts = [p.strip() for p in bw_line.split(",")]
                        if len(bw_parts) >= 3:
                            lbl_link = dict(**lbl_base, link_id=str(link_id_int))
                            tx_val = _parse_float(bw_parts[1])
                            rx_val = _parse_float(bw_parts[2])
                            # Values are in KB/s → convert to bytes/s
                            if tx_val is not None:
                                GPU_NVLINK_TX_BYTES.labels(**lbl_link).set(tx_val * 1024)
                            if rx_val is not None:
                                GPU_NVLINK_RX_BYTES.labels(**lbl_link).set(rx_val * 1024)

    def _collect_mig_metrics(self) -> None:
        """
        Collect MIG (Multi-Instance GPU) metrics.
        Only runs if any GPU has MIG mode enabled.
        Requires NVIDIA driver 450+ and A100/H100/A30 GPU.
        """
        import re as _re

        # First check which GPUs have MIG mode
        mig_mode_out = _run_cmd(
            ["nvidia-smi", "--query-gpu=index,name,uuid,mig.mode.current", "--format=csv,noheader,nounits"],
            timeout=10,
        )
        if mig_mode_out is None:
            return

        mig_enabled_gpus: List[Dict] = []
        for line in mig_mode_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_idx, gpu_name, gpu_uuid, mig_mode = parts[0], parts[1], parts[2], parts[3]
            lbl = dict(gpu_index=gpu_idx, gpu_name=gpu_name, gpu_uuid=gpu_uuid)
            enabled = 1 if mig_mode.lower() in ("enabled", "1") else 0
            try:
                GPU_MIG_MODE.labels(**lbl).set(enabled)
            except Exception:
                pass
            if enabled:
                mig_enabled_gpus.append({"index": gpu_idx, "name": gpu_name, "uuid": gpu_uuid})

        for gpu_info in mig_enabled_gpus:
            gpu_idx = gpu_info["index"]
            gpu_name = gpu_info["name"]
            gpu_uuid = gpu_info["uuid"]
            lbl_base = dict(gpu_index=gpu_idx, gpu_name=gpu_name, gpu_uuid=gpu_uuid)

            # List MIG instances
            gi_out = _run_cmd(
                ["nvidia-smi", "mig", "-lgi", "--id", gpu_idx],
                timeout=15,
            )
            instance_count = 0
            if gi_out:
                for line in gi_out.splitlines():
                    # Lines like: "  GPU  0   MIG 3g.20gb  GI ID  0   CI ID  0"
                    m = _re.search(r"MIG\s+([\w.]+)\s+GI ID\s+(\d+)\s+CI ID\s+(\d+)", line, _re.IGNORECASE)
                    if not m:
                        continue
                    profile = m.group(1)
                    gi_id = m.group(2)
                    ci_id = m.group(3)
                    instance_count += 1
                    mig_lbl = dict(
                        **lbl_base,
                        mig_gi=gi_id,
                        mig_ci=ci_id,
                        mig_profile=profile,
                    )
                    # Query per-MIG-instance metrics using nvidia-smi with MIG filter
                    mig_metrics_out = _run_cmd(
                        [
                            "nvidia-smi",
                            "--query-mig-device=gpu_instance_id,compute_instance_id,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits",
                            "--id", f"MIG-{gpu_uuid}/gi:{gi_id}/ci:{ci_id}",
                        ],
                        timeout=10,
                    )
                    if mig_metrics_out:
                        for mig_line in mig_metrics_out.strip().splitlines():
                            mig_parts = [p.strip() for p in mig_line.split(",")]
                            if len(mig_parts) >= 5:
                                try:
                                    mem_used = _parse_mb_to_bytes(mig_parts[2])
                                    mem_total = _parse_mb_to_bytes(mig_parts[3])
                                    util = _parse_float(mig_parts[4])
                                    if mem_used is not None:
                                        GPU_MIG_MEM_USED.labels(**mig_lbl).set(mem_used)
                                    if mem_total is not None:
                                        GPU_MIG_MEM_TOTAL.labels(**mig_lbl).set(mem_total)
                                    if util is not None:
                                        GPU_MIG_UTIL.labels(**mig_lbl).set(util)
                                except Exception as exc:
                                    logger.debug("Error parsing MIG instance metrics: %s", exc)

            try:
                GPU_MIG_INSTANCE_COUNT.labels(**lbl_base).set(instance_count)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_gpu_name_map(self) -> Dict[str, Dict]:
        """
        Return a dict mapping gpu_uuid → {index, name} for all GPUs.
        Falls back to empty dict on error.
        """
        out = _run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ]
        )
        mapping: Dict[str, Dict] = {}
        if not out:
            return mapping
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                idx, uuid, name = parts[0], parts[1], ",".join(parts[2:]).strip()
                mapping[uuid] = {"index": idx, "name": name}
        return mapping

    @staticmethod
    def _gpu_name_from_index(
        gpu_name_map: Dict[str, Dict], index: str
    ) -> str:
        """Reverse-lookup GPU name by index string."""
        for info in gpu_name_map.values():
            if info.get("index") == index:
                return info.get("name", "unknown")
        return "unknown"


# ---------------------------------------------------------------------------
# Background collection loop
# ---------------------------------------------------------------------------

def _run_collection_loop(collector: NvidiaCollector) -> None:
    """Continuously collect metrics every SCRAPE_INTERVAL seconds."""
    logger.info(
        "Collection loop started (interval=%ds, port=%d)",
        SCRAPE_INTERVAL,
        EXPORTER_PORT,
    )
    while True:
        start = time.monotonic()
        try:
            collector.collect()
            LAST_SCRAPE_TS.set(time.time())
        except Exception as exc:
            logger.error("Unexpected error in collection loop: %s", exc, exc_info=True)
        elapsed = time.monotonic() - start
        SCRAPE_DURATION.set(elapsed)
        sleep_time = max(0, SCRAPE_INTERVAL - elapsed)
        logger.debug("Collection took %.2fs, sleeping %.2fs", elapsed, sleep_time)
        time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# Custom HTTP handler — landing page at / and metrics at /metrics
# ---------------------------------------------------------------------------

_LANDING_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NVIDIA GPU Exporter</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0d1117;
      color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }}
    .card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 40px 48px;
      max-width: 560px;
      width: 90%;
      text-align: center;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }}
    .badge {{
      display: inline-block;
      background: #1f6feb22;
      border: 1px solid #1f6feb;
      color: #58a6ff;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 3px 10px;
      border-radius: 20px;
      margin-bottom: 18px;
    }}
    h1 {{
      font-size: 28px;
      font-weight: 700;
      color: #58a6ff;
      margin-bottom: 8px;
    }}
    .subtitle {{
      color: #8b949e;
      font-size: 14px;
      margin-bottom: 32px;
      line-height: 1.5;
    }}
    .metrics-link {{
      display: inline-block;
      background: #238636;
      color: #ffffff;
      text-decoration: none;
      padding: 12px 28px;
      border-radius: 8px;
      font-size: 15px;
      font-weight: 600;
      transition: background 0.2s;
      margin-bottom: 24px;
    }}
    .metrics-link:hover {{ background: #2ea043; }}
    .info-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 24px;
      text-align: left;
    }}
    .info-item {{
      background: #0d1117;
      border: 1px solid #21262d;
      border-radius: 8px;
      padding: 12px 14px;
    }}
    .info-label {{
      font-size: 11px;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 4px;
    }}
    .info-value {{
      font-size: 13px;
      color: #e6edf3;
      font-weight: 600;
      word-break: break-all;
    }}
    .info-value.green {{ color: #56d364; }}
    .info-value.red   {{ color: #f85149; }}
    hr {{ border: none; border-top: 1px solid #21262d; margin: 24px 0; }}
    .footer {{
      font-size: 11px;
      color: #484f58;
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">Prometheus Exporter</div>
    <h1>NVIDIA GPU Exporter</h1>
    <p class="subtitle">
      Custom nvidia-smi based GPU metrics exporter.<br>
      Supports consumer &amp; datacenter GPUs (NVLink, MIG, ECC, BAR1).
    </p>
    <a class="metrics-link" href="/metrics">/metrics</a>
    <hr>
    <div class="info-grid">
      <div class="info-item">
        <div class="info-label">Status</div>
        <div class="info-value green">{smi_status}</div>
      </div>
      <div class="info-item">
        <div class="info-label">Port</div>
        <div class="info-value">{port}</div>
      </div>
      <div class="info-item">
        <div class="info-label">Scrape Interval</div>
        <div class="info-value">{interval}s</div>
      </div>
      <div class="info-item">
        <div class="info-label">GPU Count</div>
        <div class="info-value">{gpu_count}</div>
      </div>
    </div>
    <p class="footer">nvidia-smi Prometheus Exporter &mdash; {version}</p>
  </div>
</body>
</html>
"""

_EXPORTER_VERSION = "v2.0.0-datacenter"


class _ExporterHandler(BaseHTTPRequestHandler):
    """Serves a landing page at / and Prometheus metrics at /metrics."""

    def log_message(self, fmt, *args):  # silence default access log
        logger.debug("HTTP %s %s", self.address_string(), fmt % args)

    def do_GET(self):
        if self.path in ("/", ""):
            self._serve_landing()
        elif self.path == "/metrics":
            self._serve_metrics()
        elif self.path == "/health":
            self._respond(200, "text/plain", b"OK")
        else:
            self._respond(404, "text/plain", b"404 Not Found")

    def _serve_landing(self):
        try:
            smi_ok = _nvidia_smi_available()
            smi_status = "nvidia-smi OK" if smi_ok else "nvidia-smi NOT FOUND"
            try:
                gpu_count_val = int(GPU_COUNT._value.get())
            except Exception:
                gpu_count_val = 0
            html = _LANDING_HTML.format(
                smi_status=smi_status,
                port=EXPORTER_PORT,
                interval=SCRAPE_INTERVAL,
                gpu_count=gpu_count_val,
                version=_EXPORTER_VERSION,
            )
            self._respond(200, "text/html; charset=utf-8", html.encode("utf-8"))
        except Exception as exc:
            logger.error("Error rendering landing page: %s", exc)
            self._respond(500, "text/plain", b"Internal Server Error")

    def _serve_metrics(self):
        try:
            output = generate_latest(REGISTRY)
            self._respond(200, CONTENT_TYPE_LATEST, output)
        except Exception as exc:
            logger.error("Error generating metrics: %s", exc)
            self._respond(500, "text/plain", b"Internal Server Error")

    def _respond(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_http_server(port: int) -> None:
    server = HTTPServer(("", port), _ExporterHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="http-server")
    t.start()
    logger.info("HTTP server listening on port %d  (/ = landing, /metrics = metrics)", port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("NVIDIA GPU Exporter %s starting up", _EXPORTER_VERSION)
    logger.info("Port            : %d", EXPORTER_PORT)
    logger.info("Scrape interval : %ds", SCRAPE_INTERVAL)
    logger.info("Log directory   : %s", LOG_DIR)
    logger.info("=" * 60)

    # Perform initial availability check and log the result immediately
    if _nvidia_smi_available():
        logger.info("nvidia-smi detected and working.")
        SMI_AVAILABLE.set(1)
    else:
        logger.warning(
            "nvidia-smi is NOT available on this system. "
            "The exporter will still run and expose nvidia_smi_available=0."
        )
        SMI_AVAILABLE.set(0)

    # Start the HTTP server
    _start_http_server(EXPORTER_PORT)
    logger.info("HTTP server listening on port %d", EXPORTER_PORT)

    # Start background collection in a daemon thread
    collector = NvidiaCollector()
    collection_thread = threading.Thread(
        target=_run_collection_loop,
        args=(collector,),
        daemon=True,
        name="nvidia-collector",
    )
    collection_thread.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt — shutting down.")
        sys.exit(0)


if __name__ == "__main__":
    main()
