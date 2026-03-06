#!/usr/bin/env python3
"""Test: Video generation energy measurement using AnimateDiff."""
import os, time, json, subprocess, threading
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

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

# --- AnimateDiff (SD1.5 + motion module) ---
print("🎬 Loading AnimateDiff...")
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
pipe = AnimateDiffPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
pipe = pipe.to(device)
print("   Model loaded!")

# Warm up
print("🔥 Warming up...")
with torch.no_grad():
    _ = pipe("test", num_frames=4, num_inference_steps=5, height=256, width=256)
print("   Done!")

# Test: Generate 8-frame video at 256x256, 20 steps (fits 24GB MPS)
prompt = "A beautiful sunset over mountains, timelapse, cinematic"
print(f"\n⚡ Measuring energy for 8-frame 256x256 video, 20 steps...")

pm = QuickPowerMonitor()
pm.start()
time.sleep(1)

gen_start = time.time()
with torch.no_grad():
    output = pipe(prompt, num_frames=8, num_inference_steps=20, height=256, width=256)
gen_time = time.time() - gen_start

time.sleep(1)
result = pm.stop()

# Save as GIF
frames = output.frames[0]
export_to_gif(frames, "/tmp/test_video.gif")

print(f"\n📊 Results (AnimateDiff, 8 frames, 256x256, 20 steps):")
print(f"   Generation time: {gen_time:.1f}s")
print(f"   Frames: {len(frames)}")
print(f"   Avg power: {result.get('avg_power_w','N/A')} W")
print(f"   Max power: {result.get('max_power_w','N/A')} W")
print(f"   Total energy: {result.get('total_energy_j','N/A')} J")
print(f"   Energy per frame: {result.get('total_energy_j',0)/len(frames):.1f} J/frame")

result["model"] = "animatediff-v1-5-2"
result["modality"] = "video"
result["generation_time_sec"] = round(gen_time, 2)
result["resolution"] = "256x256"
result["frames"] = len(frames)
result["steps"] = 20
result["energy_per_frame_j"] = round(result.get("total_energy_j",0)/len(frames), 2)

with open("/Users/yongjun_kim/Documents/project_energy/results/test_video.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"🎬 Video saved to /tmp/test_video.gif")
print("💾 Results saved!")
