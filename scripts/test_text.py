#!/usr/bin/env python3
"""Test: Text generation (LLM) energy measurement."""
import os, time, json, subprocess, threading
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# --- Phi-3-mini (3.8B) ---
print("📝 Loading microsoft/Phi-3-mini-4k-instruct...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,
    device_map=device
)
print(f"   Params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B on {device}")

# Warm up
print("🔥 Warming up...")
inputs = tokenizer("Hello", return_tensors="pt").to(device)
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=16)
print("   Done!")

# Test: Generate ~100 tokens
prompt = "Explain the impact of artificial intelligence on renewable energy systems in detail."
print(f"\n⚡ Measuring energy for ~256 token generation...")

pm = QuickPowerMonitor()
pm.start()
time.sleep(1)

gen_start = time.time()
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
gen_time = time.time() - gen_start
num_tokens = output.shape[1] - inputs["input_ids"].shape[1]

time.sleep(1)
result = pm.stop()

text_out = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\n📊 Results (Phi-3-mini, 256 tokens):")
print(f"   Generation time: {gen_time:.1f}s")
print(f"   Tokens generated: {num_tokens}")
print(f"   Tokens/sec: {num_tokens/gen_time:.1f}")
print(f"   Avg power: {result.get('avg_power_w','N/A')} W")
print(f"   Max power: {result.get('max_power_w','N/A')} W")
print(f"   Total energy: {result.get('total_energy_j','N/A')} J")
print(f"   Energy per token: {result.get('total_energy_j',0)/max(num_tokens,1):.2f} J/token")

result["model"] = "Phi-3-mini-4k-instruct"
result["modality"] = "text"
result["generation_time_sec"] = round(gen_time, 2)
result["tokens_generated"] = int(num_tokens)
result["tokens_per_sec"] = round(num_tokens/gen_time, 1)
result["energy_per_token_j"] = round(result.get("total_energy_j",0)/max(num_tokens,1), 4)

with open("/Users/yongjun_kim/Documents/project_energy/results/test_text.json", "w") as f:
    json.dump(result, f, indent=2)
print("💾 Saved!")
