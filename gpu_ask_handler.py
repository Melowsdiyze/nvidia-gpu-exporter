#!/usr/bin/env python3
"""
GPU Ask Handler
---------------
Handles /ask queries about GPU server load by:
1. Parsing the server IP from the question
2. Fetching all nvidia_* metrics from Prometheus
3. Sending context + question to Ollama for analysis

Usage:
    python3 gpu_ask_handler.py "gimana load gpu server 10.12.0.64"
    python3 gpu_ask_handler.py "/ask gpu server 10.12.0.22 panas ga?"

Integrate into your bot by calling: handle_gpu_ask(user_message)
"""

import re
import json
import urllib.request
import urllib.parse
from typing import Optional

# ── Config ──────────────────────────────────────────────────────────────────
PROMETHEUS_URL = "http://192.168.107.69:7000"
OLLAMA_URL     = "http://YOUR_SERVER_IP:8888"
OLLAMA_MODEL   = "gemma3:27b"
DEFAULT_PORT   = "9400"   # nvidia exporter default port

# ── Prometheus metric list to fetch ─────────────────────────────────────────
METRIC_QUERIES = {
    "nvidia-smi Status":         "nvidia_smi_available",
    "Total GPUs":                "nvidia_gpu_count",
    "GPU Utilization (%)":       "nvidia_gpu_utilization_percent",
    "Memory Utilization (%)":    "nvidia_gpu_memory_utilization_percent",
    "GPU Temperature (°C)":      "nvidia_gpu_temperature_celsius",
    "Power Draw (W)":            "nvidia_gpu_power_draw_watts",
    "Power Limit (W)":           "nvidia_gpu_power_limit_watts",
    "VRAM Used (bytes)":         "nvidia_gpu_memory_used_bytes",
    "VRAM Free (bytes)":         "nvidia_gpu_memory_free_bytes",
    "VRAM Total (bytes)":        "nvidia_gpu_memory_total_bytes",
    "Fan Speed (%)":             "nvidia_gpu_fan_speed_percent",
    "P-State":                   "nvidia_gpu_pstate",
    "Clock Graphics (MHz)":      "nvidia_gpu_clock_graphics_mhz",
    "Clock Memory (MHz)":        "nvidia_gpu_clock_memory_mhz",
    "ECC Single Bit":            "nvidia_gpu_ecc_errors_single_bit_total",
    "ECC Double Bit":            "nvidia_gpu_ecc_errors_double_bit_total",
    "Throttle SW Power Cap":     "nvidia_gpu_throttle_sw_power_cap",
    "Throttle HW Slowdown":      "nvidia_gpu_throttle_hw_slowdown",
    "Throttle HW Thermal":       "nvidia_gpu_throttle_hw_thermal_slowdown",
    "Driver Info":               "nvidia_driver_info",
    "GPU Static Info":           "nvidia_gpu_static_info",
    "Scrape Duration (s)":       "nvidia_exporter_scrape_duration_seconds",
}


# ── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah AI monitoring GPU server yang ahli dan berpengalaman.
Kamu memiliki akses ke data metrik GPU real-time dari Prometheus yang menggunakan nvidia_exporter custom.

## Tugas Utama
Saat diberi data metrik GPU dari sebuah server, kamu HARUS:
1. Menganalisis kondisi setiap GPU secara menyeluruh
2. Mendeteksi masalah atau potensi masalah
3. Memberikan ringkasan yang jelas dan actionable

## Cara Membaca Metrik

### GPU Utilization (nvidia_gpu_utilization_percent)
- 0-30% : Idle / ringan
- 30-70% : Normal / sedang dipakai
- 70-90% : Beban tinggi
- 90-100% : Fully loaded — perlu diperhatikan apakah wajar

### GPU Temperature (nvidia_gpu_temperature_celsius)
- < 60°C  : Dingin, normal
- 60-75°C : Hangat, normal untuk beban kerja
- 75-85°C : Panas, pantau terus
- > 85°C  : KRITIS — risiko throttling / kerusakan hardware

### P-State (nvidia_gpu_pstate) — Performance State
- P0 (nilai 0) : Maximum performance — GPU sedang full load
- P1-P3        : High performance
- P4-P7        : Balanced performance
- P8+          : Low power / idle
- Semakin kecil angka = semakin tinggi performa

### Power Draw vs Limit
- Draw/Limit < 50% : Beban ringan
- Draw/Limit 50-80%: Beban normal
- Draw/Limit > 90% : Hampir menyentuh power cap — bisa throttle
- Jika draw = limit : GPU sedang di-throttle oleh power cap

### VRAM Usage
- < 50% : Aman
- 50-80%: Normal
- > 80% : Penuh — perhatikan OOM risk
- 100%  : KRITIS — proses bisa crash karena out of memory

### Fan Speed
- 0%      : Passive cooling (normal untuk GPU idle)
- 30-60%  : Aktif, normal
- > 80%   : GPU sangat panas, cooling bekerja keras
- 100%    : KRITIS — overheating

### Throttle Reasons (nilai 1 = aktif)
- SW Power Cap  : Dibatasi oleh software power limit
- HW Slowdown   : Hardware sedang memperlambat karena panas/power
- HW Thermal    : Thermal protection aktif — GPU terlalu panas
- SW Thermal    : Software thermal limit aktif

### ECC Errors
- Single Bit : Error yang bisa dikoreksi — wajar sesekali
- Double Bit : SERIUS — memory GPU mungkin rusak, perlu pengecekan hardware

### nvidia-smi Status
- 1 : nvidia-smi tersedia dan GPU terdeteksi
- 0 : GPU tidak terdeteksi / driver bermasalah

### Driver Info & Static Info
- Berisi versi driver, CUDA version, VBIOS, serial number, compute capability

## Format Jawaban
Gunakan format ini saat menjawab:

### 🖥️ Server: [IP Server]
**Status Umum:** ✅ Normal / ⚠️ Perlu Perhatian / 🔴 Kritis

**GPU [index] - [nama GPU]:**
- Utilization : XX% ([status])
- Temperature : XX°C ([status])
- VRAM        : XX GB / XX GB (XX%) ([status])
- Power       : XX W / XX W (XX%) ([status])
- P-State     : PX ([penjelasan])
- Fan Speed   : XX%
- Throttle    : [Normal / Aktif - jelaskan]

**Kesimpulan:** [ringkasan kondisi server]
**Rekomendasi:** [tindakan yang perlu diambil jika ada masalah]

## Aturan Penting
- Selalu jawab dalam Bahasa Indonesia
- Jika data tidak tersedia untuk suatu metrik, sebutkan "data tidak tersedia"
- Jika ada indikasi masalah, jelaskan MENGAPA dan APA yang harus dilakukan
- Bandingkan GPU 0, GPU 1, dst. jika ada lebih dari 1 GPU
- Sebutkan apakah beban workload yang tinggi itu wajar atau perlu investigasi
"""


# ── Helper Functions ─────────────────────────────────────────────────────────

def extract_ip(text: str) -> Optional[str]:
    """Extract IP address from user message."""
    pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d+)?\b'
    match = re.search(pattern, text)
    return match.group(1) if match else None


def query_prometheus(metric: str, instance: str) -> list:
    """Query a single metric from Prometheus for a specific instance."""
    # Try with default port first, then without port
    for target in [f"{instance}:{DEFAULT_PORT}", instance]:
        query = f'{metric}{{instance="{target}"}}'
        url = f"{PROMETHEUS_URL}/api/v1/query?query={urllib.parse.quote(query)}"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                results = data.get("data", {}).get("result", [])
                if results:
                    return results
        except Exception:
            pass
    return []


def bytes_to_human(b: float) -> str:
    """Convert bytes to human-readable GB/TB."""
    gb = b / (1024 ** 3)
    if gb >= 1000:
        return f"{gb/1024:.2f} TB"
    return f"{gb:.2f} GB"


def format_metrics(ip: str, all_metrics: dict) -> str:
    """Format all fetched Prometheus metrics into a readable context string."""
    lines = [f"=== Data Metrik GPU Real-Time: Server {ip} ===\n"]

    # Group by gpu_index for GPU-specific metrics
    gpu_data = {}   # {gpu_index: {metric_name: {labels, value}}}
    server_data = {}  # non-gpu-specific metrics

    for label, results in all_metrics.items():
        if not results:
            continue
        for r in results:
            metric_labels = r.get("metric", {})
            value = r.get("value", [None, "N/A"])[1]
            gpu_idx = metric_labels.get("gpu_index", metric_labels.get("gpu", None))

            if gpu_idx is not None:
                if gpu_idx not in gpu_data:
                    gpu_data[gpu_idx] = {}
                entry = gpu_data[gpu_idx]
            else:
                entry = server_data

            # Extract useful label info
            extra = {}
            for k in ["gpu_name", "driver_version", "cuda_version", "vbios_version",
                       "serial", "compute_cap"]:
                if k in metric_labels:
                    extra[k] = metric_labels[k]

            entry[label] = {"value": value, "labels": extra}

    # Server-wide info
    if server_data:
        lines.append("[ Info Server ]")
        for name, d in server_data.items():
            v = d["value"]
            lbl = d.get("labels", {})
            extra_str = ""
            if lbl:
                extra_str = " | " + ", ".join(f"{k}={v2}" for k, v2 in lbl.items())
            lines.append(f"  {name}: {v}{extra_str}")
        lines.append("")

    # Per-GPU info
    for gpu_idx in sorted(gpu_data.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        metrics = gpu_data[gpu_idx]
        gpu_name = ""
        for m in metrics.values():
            if m.get("labels", {}).get("gpu_name"):
                gpu_name = m["labels"]["gpu_name"]
                break
        lines.append(f"[ GPU {gpu_idx} — {gpu_name} ]")

        for name, d in metrics.items():
            v = d["value"]
            # Convert bytes to human-readable
            if "bytes" in name.lower() and v not in ("N/A", "0"):
                try:
                    v = bytes_to_human(float(v))
                except ValueError:
                    pass
            # Add unit hints
            lines.append(f"  {name}: {v}")
        lines.append("")

    if not gpu_data and not server_data:
        lines.append(f"  ⚠️  Tidak ada data ditemukan untuk instance {ip}:{DEFAULT_PORT}")
        lines.append(f"  Kemungkinan: server offline, exporter tidak berjalan, atau IP salah.")

    return "\n".join(lines)


def call_ollama(system: str, user_message: str, context: str) -> str:
    """Call Ollama API with system prompt + metrics context + user question."""
    full_user = f"{context}\n\n---\nPertanyaan user: {user_message}"

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": full_user}
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,   # more factual, less creative
            "num_predict": 1024
        }
    }).encode("utf-8")

    url = f"{OLLAMA_URL}/api/chat"
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "Tidak ada respons dari Ollama.")
    except urllib.error.URLError as e:
        return f"❌ Gagal menghubungi Ollama: {e}"
    except Exception as e:
        return f"❌ Error: {e}"


# ── Main Handler ─────────────────────────────────────────────────────────────

def handle_gpu_ask(user_message: str) -> str:
    """
    Main entry point. Pass any message like:
        "/ask gimana load gpu server 10.12.0.64"
        "cek gpu 10.12.0.22 panas ga?"
        "utilization server 10.12.0.100 berapa?"
    Returns: AI analysis string
    """
    ip = extract_ip(user_message)

    if not ip:
        return (
            "❓ IP server tidak ditemukan dalam pertanyaan.\n"
            "Contoh: `/ask gimana load gpu server 10.12.0.64`"
        )

    print(f"[gpu_ask] Fetching metrics for {ip}:{DEFAULT_PORT} from Prometheus...")

    # Fetch all metrics
    all_metrics = {}
    for label, metric in METRIC_QUERIES.items():
        results = query_prometheus(metric, ip)
        all_metrics[label] = results

    # Format into context
    context = format_metrics(ip, all_metrics)
    print(f"[gpu_ask] Context built ({len(context)} chars). Calling Ollama...")

    # Call Ollama
    answer = call_ollama(SYSTEM_PROMPT, user_message, context)
    return answer


# ── CLI Usage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 gpu_ask_handler.py '<pertanyaan>'")
        print("Example: python3 gpu_ask_handler.py 'gimana load gpu server 10.12.0.64'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\n📡 Pertanyaan: {question}\n")
    print("=" * 60)
    result = handle_gpu_ask(question)
    print(result)
