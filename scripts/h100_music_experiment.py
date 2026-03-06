#!/usr/bin/env python3
"""
H100 MusicGen Experiment — MusicGen-small (589M) + MusicGen-medium (2.0B)
Run on Google Colab with H100 GPU runtime.
Copy-paste this entire script into a Colab cell and run with: !python h100_music_experiment.py
"""
import json, time, os, threading
import torch
import numpy as np

# --- GPU Power Measurement via pynvml ---
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
gpu_name = pynvml.nvmlDeviceGetName(handle)
if isinstance(gpu_name, bytes): gpu_name = gpu_name.decode()
print(f"GPU: {gpu_name}")

class PowerMonitor:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.readings = []
        self._running = False
        self._thread = None
    def start(self):
        self.readings = []
        self._running = True
        def _record():
            while self._running:
                try:
                    p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
                    self.readings.append((time.time(), p))
                except: pass
                time.sleep(self.interval)
        self._thread = threading.Thread(target=_record, daemon=True)
        self._thread.start()
        time.sleep(0.3)
    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=3)
        if len(self.readings) < 2:
            return {'total_energy_j':0,'avg_power_w':0,'peak_power_w':0,'samples':0}
        powers = [r[1] for r in self.readings]
        energy = sum((self.readings[i][1]+self.readings[i-1][1])/2*(self.readings[i][0]-self.readings[i-1][0])
                      for i in range(1, len(self.readings)))
        return {
            'total_energy_j': round(float(energy), 3),
            'avg_power_w': round(float(np.mean(powers)), 3),
            'peak_power_w': round(float(np.max(powers)), 3),
            'samples': len(powers)
        }

PROMPT = "happy rock song with electric guitar and drums"
N_REPEATS = 30
results = []
pm = PowerMonitor()

# ===== MusicGen-small (589M) =====
print("\n" + "="*60)
print("MusicGen-small (589M) — 3 configs × 30 repeats")
print("="*60)

from transformers import AutoProcessor, MusicgenForConditionalGeneration
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model_small = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to("cuda")
model_small.eval()

# Warmup
inputs = processor(text=[PROMPT], padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    model_small.generate(**inputs, max_new_tokens=32)
torch.cuda.synchronize()
print("Warmup done", flush=True)

for max_tok in [128, 256, 512]:
    print(f"\nsmall / {max_tok} tokens:", flush=True)
    for rep in range(N_REPEATS):
        inputs = processor(text=[PROMPT], padding=True, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        pm.start()
        t0 = time.time()
        with torch.no_grad():
            out = model_small.generate(**inputs, max_new_tokens=max_tok)
        torch.cuda.synchronize()
        gen_time = time.time() - t0
        power = pm.stop()
        
        actual_tokens = out.shape[-1]
        record = {
            'model': 'musicgen-small',
            'params': '589M',
            'max_tokens': max_tok,
            'actual_tokens': int(actual_tokens),
            'prompt': PROMPT,
            'run': rep,
            'seed': rep,
            'generation_time_s': round(gen_time, 3),
            'gpu': gpu_name,
            **power
        }
        results.append(record)
        print(f"  {rep+1}/{N_REPEATS}: {power['total_energy_j']:.1f}J {gen_time:.1f}s {actual_tokens}tok", flush=True)

del model_small
torch.cuda.empty_cache()
import gc; gc.collect()

# ===== MusicGen-medium (2.0B) =====
print("\n" + "="*60)
print("MusicGen-medium (2.0B) — 2 configs × 30 repeats")
print("="*60)

model_medium = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium").to("cuda")
model_medium.eval()

# Warmup
inputs = processor(text=[PROMPT], padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    model_medium.generate(**inputs, max_new_tokens=32)
torch.cuda.synchronize()
print("Warmup done", flush=True)

for max_tok in [256, 512]:
    print(f"\nmedium / {max_tok} tokens:", flush=True)
    for rep in range(N_REPEATS):
        inputs = processor(text=[PROMPT], padding=True, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        pm.start()
        t0 = time.time()
        with torch.no_grad():
            out = model_medium.generate(**inputs, max_new_tokens=max_tok)
        torch.cuda.synchronize()
        gen_time = time.time() - t0
        power = pm.stop()
        
        actual_tokens = out.shape[-1]
        record = {
            'model': 'musicgen-medium',
            'params': '2.0B',
            'max_tokens': max_tok,
            'actual_tokens': int(actual_tokens),
            'prompt': PROMPT,
            'run': rep,
            'seed': rep,
            'generation_time_s': round(gen_time, 3),
            'gpu': gpu_name,
            **power
        }
        results.append(record)
        print(f"  {rep+1}/{N_REPEATS}: {power['total_energy_j']:.1f}J {gen_time:.1f}s {actual_tokens}tok", flush=True)

# ===== Save & Summary =====
outfile = "music_musicgen_H100.json"
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Saved {len(results)} records to {outfile}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for model_name in ['musicgen-small', 'musicgen-medium']:
    d = [r for r in results if r['model'] == model_name and r.get('total_energy_j', 0) > 0]
    if not d: continue
    print(f"\n{model_name}:")
    for mt in sorted(set(r['max_tokens'] for r in d)):
        vals = [r['total_energy_j'] for r in d if r['max_tokens'] == mt]
        times = [r['generation_time_s'] for r in d if r['max_tokens'] == mt]
        print(f"  {mt} tokens: {np.mean(vals):.1f} ± {np.std(vals):.1f}J, {np.mean(times):.1f}s (CV={np.std(vals)/np.mean(vals)*100:.1f}%)")

print(f"\nTotal: {len(results)} records")
print("Copy music_musicgen_H100.json and send to me!")
