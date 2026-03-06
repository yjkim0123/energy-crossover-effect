#!/usr/bin/env python3
"""Test: Image generation (Stable Diffusion) energy measurement."""
import os, time, json, subprocess, threading
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from diffusers import StableDiffusionPipeline

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

device = "mps"

# --- Stable Diffusion v1.5 (open, no auth needed) ---
print("🖼️ Loading Stable Diffusion v1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to(device)
param_count = sum(p.numel() for p in pipe.unet.parameters()) + sum(p.numel() for p in pipe.text_encoder.parameters())
print(f"   UNet+TextEnc params: {param_count/1e9:.1f}B on {device}")

# Warm up
print("🔥 Warming up...")
with torch.no_grad():
    _ = pipe("test", num_inference_steps=5, height=256, width=256)
print("   Done!")

# Test: Generate 512x512 image, 30 steps
prompt = "A beautiful sunset over mountains with a river in the foreground, photorealistic"
print(f"\n⚡ Measuring energy for 512x512 image, 30 steps...")

pm = QuickPowerMonitor()
pm.start()
time.sleep(1)

gen_start = time.time()
with torch.no_grad():
    result_img = pipe(prompt, num_inference_steps=30, height=512, width=512)
gen_time = time.time() - gen_start

time.sleep(1)
result = pm.stop()

# Save image
result_img.images[0].save("/tmp/test_sd.png")

print(f"\n📊 Results (SD 2.1, 512x512, 30 steps):")
print(f"   Generation time: {gen_time:.1f}s")
print(f"   Avg power: {result.get('avg_power_w','N/A')} W")
print(f"   Max power: {result.get('max_power_w','N/A')} W")
print(f"   Total energy: {result.get('total_energy_j','N/A')} J")

result["model"] = "stable-diffusion-v1-5"
result["modality"] = "image"
result["generation_time_sec"] = round(gen_time, 2)
result["resolution"] = "512x512"
result["steps"] = 30

with open("/Users/yongjun_kim/Documents/project_energy/results/test_image.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"🖼️ Image saved to /tmp/test_sd.png")
print("💾 Results saved!")
