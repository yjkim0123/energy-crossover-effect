#!/usr/bin/env python3
"""Quick test: MusicGen on MPS + power measurement."""
import os, time, json, subprocess, threading
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# --- Simple power monitor ---
class QuickPowerMonitor:
    def __init__(self):
        self.readings = []
        self._running = False
        
    def start(self):
        self.readings = []
        self._running = True
        self._start = time.time()
        self._proc = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "cpu_power", "-i", "500"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        def _read():
            for line in self._proc.stdout:
                if not self._running: break
                if "Combined Power" in line:
                    try:
                        mw = float(line.split(":")[1].replace("mW","").strip())
                        self.readings.append((time.time()-self._start, mw/1000))
                    except: pass
        self._thread = threading.Thread(target=_read, daemon=True)
        self._thread.start()
        
    def stop(self):
        self._running = False
        self._proc.terminate()
        self._proc.wait()
        if not self.readings: return {}
        powers = [r[1] for r in self.readings]
        duration = self.readings[-1][0] - self.readings[0][0]
        # Trapezoidal energy
        energy_j = sum((self.readings[i][1]+self.readings[i-1][1])/2 * 
                       (self.readings[i][0]-self.readings[i-1][0]) 
                       for i in range(1, len(self.readings)))
        return {
            "duration_sec": round(duration, 2),
            "samples": len(self.readings),
            "avg_power_w": round(sum(powers)/len(powers), 2),
            "max_power_w": round(max(powers), 2),
            "total_energy_j": round(energy_j, 2),
            "energy_kwh": round(energy_j/3600000, 8)
        }

# --- Main ---
print("🎵 Loading MusicGen-small...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"   Device: {device}")

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model = model.to(device)
print(f"   Model loaded! Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

# Warm up
print("🔥 Warming up...")
inputs = processor(text=["happy rock song"], padding=True, return_tensors="pt").to(device)
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=64)
print("   Warm up done!")

# Actual test
print("\n⚡ Measuring energy for 5s music generation...")
pm = QuickPowerMonitor()
pm.start()
time.sleep(1)  # baseline

gen_start = time.time()
inputs = processor(text=["upbeat electronic dance music"], padding=True, return_tensors="pt").to(device)
with torch.no_grad():
    audio = model.generate(**inputs, max_new_tokens=256)  # ~5 seconds of audio
gen_time = time.time() - gen_start

time.sleep(1)  # tail
result = pm.stop()

print(f"\n📊 Results:")
print(f"   Generation time: {gen_time:.1f}s")
print(f"   Audio shape: {audio.shape}")
print(f"   Avg power: {result.get('avg_power_w', 'N/A')} W")
print(f"   Max power: {result.get('max_power_w', 'N/A')} W")  
print(f"   Total energy: {result.get('total_energy_j', 'N/A')} J")
print(f"   Energy (kWh): {result.get('energy_kwh', 'N/A')}")

# Save audio
import scipy.io.wavfile as wav
sampling_rate = model.config.audio_encoder.sampling_rate
audio_np = audio[0, 0].cpu().numpy()
wav.write("/tmp/test_musicgen.wav", sampling_rate, audio_np)
print(f"\n🎵 Audio saved to /tmp/test_musicgen.wav ({len(audio_np)/sampling_rate:.1f}s)")

# Save result
result["model"] = "musicgen-small"
result["generation_time_sec"] = round(gen_time, 2)
result["audio_length_sec"] = round(len(audio_np)/sampling_rate, 1)
with open("/Users/yongjun_kim/Documents/project_energy/results/test_musicgen.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"💾 Results saved!")
