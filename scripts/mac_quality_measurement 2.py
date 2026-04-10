#!/usr/bin/env python3
"""Measure image quality (CLIP score) + energy on Mac for SD v1.5 with varying steps."""
import torch, time, gc, json, os, subprocess, tempfile, signal
import numpy as np
from diffusers import StableDiffusionPipeline
import open_clip
from PIL import Image

DEVICE = "mps"
REPEATS = 15  # fewer repeats, more configs
RESULTS_DIR = os.path.expanduser("~/Documents/project_energy/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Power monitoring (file-based)
class PowerMonitor:
    def __init__(self):
        self._proc = None
        self._tmpfile = None

    def start(self):
        self._tmpfile = tempfile.mktemp(prefix='pm_', suffix='.txt')
        self._proc = subprocess.Popen(
            ['powermetrics', '-i', '100', '-n', '9999', '--samplers', 'cpu_power'],
            stdout=open(self._tmpfile, 'w'), stderr=subprocess.DEVNULL
        )
        time.sleep(0.2)

    def stop(self):
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except:
                try: self._proc.kill()
                except: pass
        readings = []
        if self._tmpfile and os.path.exists(self._tmpfile):
            try:
                with open(self._tmpfile) as f:
                    for line in f:
                        if 'Combined Power' in line:
                            try:
                                raw = line.split(':')[1].strip()
                                val = float(raw.replace('mW','').strip()) / 1000.0 if 'mW' in raw else float(raw.replace('W','').strip())
                                readings.append(val)
                            except: pass
            except: pass
            try: os.unlink(self._tmpfile)
            except: pass
        if len(readings) >= 2:
            powers = np.array(readings)
            total_e = float(np.sum(powers) * 0.1)
            return {'total_energy_j': round(total_e,2), 'avg_power_w': round(float(np.mean(powers)),2),
                    'max_power_w': round(float(np.max(powers)),2), 'samples': len(powers)}
        return {}

# Test power
print("Testing powermetrics...", flush=True)
pm_test = PowerMonitor()
pm_test.start()
time.sleep(2)
test = pm_test.stop()
if 'total_energy_j' not in test:
    print("FAIL: Run with sudo python3", flush=True)
    exit(1)
print(f"  OK: {test['avg_power_w']:.1f} W", flush=True)

# Load CLIP
print("Loading CLIP model...", flush=True)
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.eval()
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

def get_clip_score(image, text):
    img_input = clip_preprocess(image).unsqueeze(0)
    txt_input = clip_tokenizer([text])
    with torch.no_grad():
        img_feat = clip_model.encode_image(img_input)
        txt_feat = clip_model.encode_text(txt_input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        score = (img_feat @ txt_feat.T).item()
    return round(score, 4)

# Load SD v1.5
print("Loading SD v1.5...", flush=True)
pipe = StableDiffusionPipeline.from_pretrained(
    'stable-diffusion-v1-5/stable-diffusion-v1-5',
    torch_dtype=torch.float16
).to(DEVICE)

prompt = 'A serene landscape with mountains and a lake at sunset, highly detailed, 8k'

results = []
for res in [128, 256, 384, 512]:
    for steps in [5, 10, 15, 20, 30, 50]:
        print(f"SD v1.5 {res}x{res} {steps}steps", flush=True)
        # warmup
        _ = pipe(prompt, height=res, width=res, num_inference_steps=steps,
                guidance_scale=7.5, generator=torch.Generator(DEVICE).manual_seed(0))

        for run in range(1, REPEATS+1):
            torch.mps.synchronize()
            pm = PowerMonitor()
            pm.start()
            t0 = time.time()
            output = pipe(prompt, height=res, width=res, num_inference_steps=steps,
                    guidance_scale=7.5, generator=torch.Generator(DEVICE).manual_seed(run))
            torch.mps.synchronize()
            gen_time = time.time() - t0
            stats = pm.stop()
            clip_s = get_clip_score(output.images[0], prompt)
            stats.update({
                'generation_time_sec': round(gen_time, 2),
                'resolution': f'{res}x{res}',
                'steps': steps,
                'clip_score': clip_s,
                'modality': 'image',
                'model': 'SD-v1-5',
                'params_B': 0.9,
                'run': run,
                'hardware': 'Apple M4 Pro',
                'backend': 'mps'
            })
            results.append(stats)
            if run % 5 == 0:
                e = stats.get('total_energy_j', 'N/A')
                print(f"  Run {run}/{REPEATS}: {e}J, CLIP={clip_s:.3f}", flush=True)

        # Save incrementally
        outpath = os.path.join(RESULTS_DIR, 'quality_image_sd15_Mac.json')
        with open(outpath, 'w') as f:
            json.dump(results, f, indent=2)

outpath = os.path.join(RESULTS_DIR, 'quality_image_sd15_Mac.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
valid = sum(1 for r in results if 'total_energy_j' in r)
print(f"\n✅ Done! {len(results)} records ({valid} with energy) → {outpath}", flush=True)
