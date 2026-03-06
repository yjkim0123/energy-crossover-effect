#!/usr/bin/env python3
"""Quick test: MusicGen-medium energy (128 tokens only, safe for 24GB)."""
import os, time, json, subprocess, threading
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

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
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
        self._proc.terminate(); self._proc.wait()
        if not self.readings: return {}
        powers = [r[1] for r in self.readings]
        duration = self.readings[-1][0] - self.readings[0][0]
        energy_j = sum((self.readings[i][1]+self.readings[i-1][1])/2 *
                       (self.readings[i][0]-self.readings[i-1][0])
                       for i in range(1, len(self.readings)))
        return {"duration_sec": round(duration,2), "samples": len(self.readings),
                "avg_power_w": round(sum(powers)/len(powers),2),
                "max_power_w": round(max(powers),2),
                "total_energy_j": round(energy_j,2)}

device = "mps"
prompt = "upbeat electronic dance music with a catchy melody"
REPEATS = 5

print("🎵 Loading MusicGen-medium (1.5B)...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
model = model.to(device)
params = sum(p.numel() for p in model.parameters())
print(f"   Params: {params/1e6:.0f}M on {device}")

# Warm up
inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=32)
print("   Warmed up!")

results = []
for tokens, label in [(128, "~2.5s"), (256, "~5s")]:
    print(f"\n📏 medium × {label} ({tokens} tokens) × {REPEATS} runs")
    for run in range(REPEATS):
        pm = QuickPowerMonitor()
        pm.start()
        time.sleep(1)
        
        gen_start = time.time()
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            audio = model.generate(**inputs, max_new_tokens=tokens)
        gen_time = time.time() - gen_start
        
        time.sleep(1)
        result = pm.stop()
        audio_len = audio.shape[-1] / model.config.audio_encoder.sampling_rate
        
        result["model"] = "medium"
        result["params_M"] = round(params/1e6, 0)
        result["max_tokens"] = tokens
        result["audio_sec"] = round(audio_len, 1)
        result["gen_time_sec"] = round(gen_time, 2)
        result["run"] = run + 1
        results.append(result)
        
        print(f"  Run {run+1}: {result.get('total_energy_j',0):.0f}J | "
              f"{result.get('avg_power_w',0):.1f}W | {gen_time:.1f}s | audio={audio_len:.1f}s")

with open("/Users/yongjun_kim/Documents/project_energy/results/test_music_medium.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n💾 Saved!")
