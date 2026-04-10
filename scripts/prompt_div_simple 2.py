#!/usr/bin/env python3
"""Simple prompt diversity experiment — text only first, then image."""
import json, time, os, subprocess, signal, re, sys
import torch
import numpy as np

RESULTS_DIR = os.path.expanduser("~/Documents/project_energy/results")

class PM:
    """Power measurement using powermetrics with thread-based reading (matches original experiment)."""
    def __init__(self):
        self.proc = None
        self.readings = []
        self._running = False
        self._thread = None
    def start(self):
        import threading
        self.readings = []
        self._running = True
        self.proc = subprocess.Popen(
            ["sudo", "powermetrics", "-n", "9999", "--samplers", "cpu_power", "-i", "100"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        def reader():
            t0 = time.time()
            for line in self.proc.stdout:
                if not self._running: break
                m = re.search(r'Combined Power.*?:\s*([\d.]+)\s*mW', line)
                if m:
                    self.readings.append((time.time()-t0, float(m.group(1))/1000.0))
        self._thread = threading.Thread(target=reader, daemon=True)
        self._thread.start()
        time.sleep(0.3)
    def stop(self):
        self._running = False
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None
        if self._thread:
            self._thread.join(timeout=3)
        if not self.readings or len(self.readings) < 2:
            return {'avg_power_w':0,'peak_power_w':0,'total_energy_j':0,'samples':0}
        powers = [r[1] for r in self.readings]
        e = sum((self.readings[i][1]+self.readings[i-1][1])/2*(self.readings[i][0]-self.readings[i-1][0])
                for i in range(1,len(self.readings)))
        return {'avg_power_w':round(float(np.mean(powers)),3),'peak_power_w':round(float(np.max(powers)),3),
                'total_energy_j':round(float(e),3),'samples':len(powers)}

PROMPTS_TEXT = [
    "Write a short story about a robot learning to paint.",
    "Explain the theory of general relativity in simple terms.",
    "Write a poem about the ocean at midnight.",
    "Describe the process of photosynthesis step by step.",
    "Tell me a funny joke about a programmer and a rubber duck.",
]

PROMPTS_IMAGE = [
    "A serene landscape with mountains and a lake at sunset, highly detailed, 8k.",
    "A cyberpunk city street at night with neon signs and rain, photorealistic.",
    "A cute orange tabby cat sleeping on a stack of books, watercolor style.",
    "An astronaut floating in space with Earth in the background, cinematic.",
    "A medieval castle on a cliff overlooking the sea, dramatic lighting.",
]

results = []
pm = PM()

# ===== TEXT =====
print("===== TEXT: 5 prompts × 10 repeats =====", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
mdl = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.float16, device_map="mps")
mdl.eval()
# warmup
inp = tok("Hello", return_tensors="pt").to("mps")
with torch.no_grad(): mdl.generate(**inp, max_new_tokens=10, do_sample=True, temperature=0.7)
torch.mps.synchronize()
print("Warmup done", flush=True)

for pi, prompt in enumerate(PROMPTS_TEXT):
    print(f"Text prompt {pi+1}/5: '{prompt[:40]}...'", flush=True)
    for rep in range(10):
        inp = tok(prompt, return_tensors="pt").to("mps")
        torch.mps.synchronize()
        pm.start()
        t0 = time.time()
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=128, do_sample=True, temperature=0.7)
        torch.mps.synchronize()
        gt = time.time()-t0
        pw = pm.stop()
        ntok = out.shape[1]-inp['input_ids'].shape[1]
        r = {'modality':'text','prompt_id':pi,'prompt':prompt,'max_tokens':128,
             'actual_tokens':int(ntok),'run':rep,'generation_time_s':round(gt,3)}
        r.update(pw)
        results.append(r)
        print(f"  {rep+1}/10: {pw['total_energy_j']:.1f}J {gt:.1f}s {ntok}tok", flush=True)

del mdl, tok
torch.mps.empty_cache()
import gc; gc.collect()
print(f"Text done: {len(results)} records", flush=True)

# ===== IMAGE =====
print("\n===== IMAGE: 5 prompts × 10 repeats =====", flush=True)
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("mps")
pipe.set_progress_bar_config(disable=True)
# warmup
_ = pipe("warmup", num_inference_steps=2, height=256, width=256, generator=torch.Generator("mps").manual_seed(0))
torch.mps.synchronize()
print("Image warmup done", flush=True)

for pi, prompt in enumerate(PROMPTS_IMAGE):
    print(f"Image prompt {pi+1}/5: '{prompt[:40]}...'", flush=True)
    for rep in range(10):
        gen = torch.Generator("mps").manual_seed(rep)
        torch.mps.synchronize()
        pm.start()
        t0 = time.time()
        _ = pipe(prompt, num_inference_steps=20, height=512, width=512, guidance_scale=7.5, generator=gen)
        torch.mps.synchronize()
        gt = time.time()-t0
        pw = pm.stop()
        r = {'modality':'image','prompt_id':pi,'prompt':prompt,'resolution':'512x512',
             'steps':20,'run':rep,'generation_time_s':round(gt,3)}
        r.update(pw)
        results.append(r)
        print(f"  {rep+1}/10: {pw['total_energy_j']:.1f}J {gt:.1f}s", flush=True)

# Save
out = os.path.join(RESULTS_DIR, "prompt_diversity_Mac.json")
with open(out,'w') as f: json.dump(results, f, indent=2)
print(f"\n✅ Saved {len(results)} records to {out}", flush=True)

# Summary
print("\n===== SUMMARY =====", flush=True)
for mod in ['text','image']:
    d = [r for r in results if r['modality']==mod and r.get('total_energy_j',0)>0]
    if not d: continue
    print(f"\n{mod.upper()}:", flush=True)
    pmeans = []
    for pi in range(5):
        vals = [r['total_energy_j'] for r in d if r['prompt_id']==pi]
        if vals:
            pmeans.append(np.mean(vals))
            print(f"  P{pi+1}: {np.mean(vals):.1f}±{np.std(vals):.1f}J (CV={np.std(vals)/np.mean(vals)*100:.1f}%)", flush=True)
    bvar = np.var(pmeans)
    wvars = [np.var([r['total_energy_j'] for r in d if r['prompt_id']==pi]) for pi in range(5)]
    wvar = np.mean(wvars)
    print(f"  Between-prompt var: {bvar:.2f}, Within-prompt var: {wvar:.2f}", flush=True)
    print(f"  Ratio: {bvar/wvar:.3f}" if wvar>0 else "", flush=True)
    all_e = [r['total_energy_j'] for r in d]
    print(f"  Overall: {np.mean(all_e):.1f}±{np.std(all_e):.1f}J (CV={np.std(all_e)/np.mean(all_e)*100:.1f}%)", flush=True)
