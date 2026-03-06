#!/usr/bin/env python3
"""Experiment: MusicGen model size scaling (small/medium/large) × audio length."""
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
                "total_energy_j": round(energy_j,2),
                "energy_kwh": round(energy_j/3600000,8)}

device = "mps" if torch.backends.mps.is_available() else "cpu"
prompt = "upbeat electronic dance music with a catchy melody"
REPEATS = 5  # 5 repeats per config for now

# Token lengths → approximate audio seconds
# MusicGen: ~50 tokens/sec at 32kHz
token_configs = [
    (128, "~2.5s"),
    (256, "~5s"),
    (512, "~10s"),
]

model_configs = [
    ("facebook/musicgen-small", "small", "300M"),
    ("facebook/musicgen-medium", "medium", "1.5B"),
]
# large (3.3B) might OOM on 24GB, try if medium works

all_results = []

for model_name, model_label, param_str in model_configs:
    print(f"\n{'='*60}")
    print(f"🎵 Loading {model_label} ({param_str})...")
    print(f"{'='*60}")
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"   Params: {actual_params/1e6:.0f}M on {device}")
    
    # Warm up
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=64)
    print("   Warmed up!")
    
    for max_tokens, length_label in token_configs:
        print(f"\n  📏 {model_label} × {length_label} ({max_tokens} tokens) × {REPEATS} runs")
        
        for run in range(REPEATS):
            pm = QuickPowerMonitor()
            pm.start()
            time.sleep(1)
            
            gen_start = time.time()
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                audio = model.generate(**inputs, max_new_tokens=max_tokens)
            gen_time = time.time() - gen_start
            
            time.sleep(1)
            result = pm.stop()
            
            audio_len = audio.shape[-1] / model.config.audio_encoder.sampling_rate
            
            result["model"] = model_label
            result["params"] = param_str
            result["actual_params_M"] = round(actual_params/1e6, 0)
            result["max_tokens"] = max_tokens
            result["target_length"] = length_label
            result["actual_audio_sec"] = round(audio_len, 1)
            result["generation_time_sec"] = round(gen_time, 2)
            result["run"] = run + 1
            result["modality"] = "music"
            
            all_results.append(result)
            print(f"    Run {run+1}: {result.get('total_energy_j',0):.0f}J | "
                  f"{result.get('avg_power_w',0):.1f}W | {gen_time:.1f}s | "
                  f"audio={audio_len:.1f}s")
        
        # Summary for this config
        config_results = [r for r in all_results 
                         if r["model"]==model_label and r["max_tokens"]==max_tokens]
        energies = [r["total_energy_j"] for r in config_results]
        avg_e = sum(energies)/len(energies)
        print(f"    📊 Mean: {avg_e:.0f}J")
    
    # Free memory
    del model, processor
    torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
    import gc; gc.collect()
    print(f"   🗑️ {model_label} unloaded")

# Save all
outpath = "/Users/yongjun_kim/Documents/project_energy/results/exp_music_scaling.json"
with open(outpath, "w") as f:
    json.dump(all_results, f, indent=2)

# Final summary table
print(f"\n{'='*60}")
print("📊 MUSIC SCALING SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<10} {'Length':<8} {'Energy(J)':<12} {'Power(W)':<10} {'Time(s)':<10}")
for model_label, _, _ in model_configs:
    for max_tokens, length_label in token_configs:
        rs = [r for r in all_results if r["model"]==model_label and r["max_tokens"]==max_tokens]
        if rs:
            avg_e = sum(r["total_energy_j"] for r in rs)/len(rs)
            avg_p = sum(r["avg_power_w"] for r in rs)/len(rs)
            avg_t = sum(r["generation_time_sec"] for r in rs)/len(rs)
            print(f"{model_label:<10} {length_label:<8} {avg_e:<12.1f} {avg_p:<10.1f} {avg_t:<10.1f}")

print(f"\n💾 Saved to {outpath}")
