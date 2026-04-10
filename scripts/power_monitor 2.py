#!/usr/bin/env python3
"""
Power Monitor for Apple Silicon Mac
Measures CPU + GPU + ANE power consumption during AI inference.
Uses macOS powermetrics (requires sudo).
"""
import subprocess
import time
import json
import threading
import os
from datetime import datetime


class PowerMonitor:
    """Real-time power monitoring for Apple Silicon."""
    
    def __init__(self, interval_ms=500):
        self.interval_ms = interval_ms
        self.measurements = []
        self._running = False
        self._process = None
        self._thread = None
        self.baseline_power = None  # watts
    
    def measure_baseline(self, duration_sec=10):
        """Measure baseline (idle) power for given duration."""
        print(f"⏳ Measuring baseline power for {duration_sec}s (keep system idle)...")
        readings = []
        proc = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "cpu_power",
             "-i", str(self.interval_ms), "-n", str(int(duration_sec * 1000 / self.interval_ms))],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for line in proc.stdout:
            line = line.strip()
            if "Combined Power" in line:
                try:
                    mw = float(line.split(":")[1].strip().replace("mW", "").strip())
                    readings.append(mw / 1000.0)  # Convert to Watts
                except (ValueError, IndexError):
                    pass
        proc.wait()
        if readings:
            self.baseline_power = sum(readings) / len(readings)
            print(f"✅ Baseline power: {self.baseline_power:.2f} W ({len(readings)} samples)")
        else:
            self.baseline_power = 5.0  # fallback
            print(f"⚠️ No baseline readings, using fallback: {self.baseline_power} W")
        return self.baseline_power
    
    def start(self):
        """Start continuous power monitoring in background."""
        self.measurements = []
        self._running = True
        self._start_time = time.time()
        
        self._process = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "cpu_power",
             "-i", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        def _reader():
            cpu_power = gpu_power = ane_power = combined = None
            for line in self._process.stdout:
                if not self._running:
                    break
                line = line.strip()
                if "CPU Power:" in line and "Combined" not in line:
                    try:
                        cpu_power = float(line.split(":")[1].replace("mW", "").strip()) / 1000.0
                    except:
                        pass
                elif "GPU Power:" in line:
                    try:
                        gpu_power = float(line.split(":")[1].replace("mW", "").strip()) / 1000.0
                    except:
                        pass
                elif "ANE Power:" in line:
                    try:
                        ane_power = float(line.split(":")[1].replace("mW", "").strip()) / 1000.0
                    except:
                        pass
                elif "Combined Power" in line:
                    try:
                        combined = float(line.split(":")[1].replace("mW", "").strip()) / 1000.0
                        self.measurements.append({
                            "timestamp": time.time(),
                            "elapsed_sec": time.time() - self._start_time,
                            "cpu_w": cpu_power or 0,
                            "gpu_w": gpu_power or 0,
                            "ane_w": ane_power or 0,
                            "combined_w": combined
                        })
                    except:
                        pass
        
        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()
        print("🔴 Power monitoring started")
    
    def stop(self):
        """Stop monitoring and return summary."""
        self._running = False
        if self._process:
            self._process.terminate()
            self._process.wait()
        if self._thread:
            self._thread.join(timeout=3)
        
        total_time = time.time() - self._start_time
        print(f"⏹️ Power monitoring stopped ({total_time:.1f}s, {len(self.measurements)} samples)")
        return self.get_summary()
    
    def get_summary(self):
        """Calculate energy summary from measurements."""
        if not self.measurements:
            return {"error": "No measurements"}
        
        powers = [m["combined_w"] for m in self.measurements]
        avg_power = sum(powers) / len(powers)
        max_power = max(powers)
        min_power = min(powers)
        
        # Total time
        total_sec = self.measurements[-1]["elapsed_sec"] - self.measurements[0]["elapsed_sec"]
        if total_sec <= 0:
            total_sec = len(self.measurements) * (self.interval_ms / 1000.0)
        
        # Energy = integral of power over time (trapezoid)
        total_energy_j = 0
        for i in range(1, len(self.measurements)):
            dt = self.measurements[i]["elapsed_sec"] - self.measurements[i-1]["elapsed_sec"]
            avg_p = (self.measurements[i]["combined_w"] + self.measurements[i-1]["combined_w"]) / 2
            total_energy_j += avg_p * dt
        
        # Subtract baseline
        baseline_energy_j = (self.baseline_power or 0) * total_sec
        net_energy_j = total_energy_j - baseline_energy_j
        
        # Convert
        total_energy_kwh = total_energy_j / 3600000
        net_energy_kwh = net_energy_j / 3600000
        
        # CO2 (global average: 0.475 kgCO2/kWh, IEA 2023)
        co2_g = net_energy_kwh * 475  # grams
        
        summary = {
            "duration_sec": round(total_sec, 2),
            "num_samples": len(self.measurements),
            "avg_power_w": round(avg_power, 2),
            "max_power_w": round(max_power, 2),
            "min_power_w": round(min_power, 2),
            "baseline_power_w": round(self.baseline_power or 0, 2),
            "total_energy_j": round(total_energy_j, 2),
            "net_energy_j": round(net_energy_j, 2),
            "total_energy_kwh": round(total_energy_kwh, 6),
            "net_energy_kwh": round(net_energy_kwh, 6),
            "co2_grams": round(co2_g, 4),
            "gpu_avg_w": round(sum(m["gpu_w"] for m in self.measurements) / len(self.measurements), 2),
            "cpu_avg_w": round(sum(m["cpu_w"] for m in self.measurements) / len(self.measurements), 2),
            "ane_avg_w": round(sum(m["ane_w"] for m in self.measurements) / len(self.measurements), 2),
        }
        return summary
    
    def save_raw(self, filepath):
        """Save raw measurements to JSON."""
        with open(filepath, 'w') as f:
            json.dump({
                "metadata": {
                    "machine": "Mac mini M4 Pro 24GB",
                    "baseline_power_w": self.baseline_power,
                    "interval_ms": self.interval_ms,
                    "recorded_at": datetime.now().isoformat()
                },
                "measurements": self.measurements
            }, f, indent=2)
        print(f"💾 Raw data saved to {filepath}")


def run_experiment(name, task_fn, monitor, repeats=30, output_dir="results"):
    """Run a generation task multiple times and measure energy."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    print(f"\n{'='*60}")
    print(f"🧪 Experiment: {name} ({repeats} repeats)")
    print(f"{'='*60}")
    
    for i in range(repeats):
        print(f"\n  Run {i+1}/{repeats}...")
        time.sleep(2)  # cool down between runs
        
        monitor.start()
        
        # Run the generation task
        gen_start = time.time()
        output_info = task_fn()
        gen_time = time.time() - gen_start
        
        time.sleep(1)  # capture tail power
        summary = monitor.stop()
        summary["run"] = i + 1
        summary["generation_time_sec"] = round(gen_time, 2)
        if output_info:
            summary.update(output_info)
        results.append(summary)
        
        print(f"  ⚡ Energy: {summary.get('net_energy_j', 0):.1f} J | "
              f"Power: {summary.get('avg_power_w', 0):.1f} W | "
              f"Time: {gen_time:.1f}s")
    
    # Save results
    filepath = os.path.join(output_dir, f"{name}.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    energies = [r["net_energy_j"] for r in results]
    avg_e = sum(energies) / len(energies)
    std_e = (sum((e - avg_e)**2 for e in energies) / len(energies)) ** 0.5
    print(f"\n📊 {name} Summary:")
    print(f"   Mean energy: {avg_e:.1f} ± {std_e:.1f} J")
    print(f"   Mean power:  {sum(r['avg_power_w'] for r in results)/len(results):.1f} W")
    print(f"   Saved to: {filepath}")
    
    return results


if __name__ == "__main__":
    # Quick test
    pm = PowerMonitor(interval_ms=500)
    baseline = pm.measure_baseline(duration_sec=5)
    
    print("\n--- Test: 10s idle measurement ---")
    pm.start()
    time.sleep(10)
    summary = pm.stop()
    
    print("\n📊 Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
