#!/usr/bin/env python3
"""
Prompt diversity experiment: measure energy variation across different prompts.
Tests 5 prompts × 10 repeats for text (Phi-3, 128 tokens) and image (SD v1.5, 512x512, 20 steps).
Goal: Show that energy is prompt-invariant (dominated by config, not content).
"""
import json, time, os, subprocess, signal, re, sys
import torch
import numpy as np
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

RESULTS_DIR = os.path.expanduser("~/Documents/project_energy/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Power measurement (same as main experiment) ---
class PowerMeasurement:
    def __init__(self):
        self.process = None
        self.tmpfile = f"/tmp/powermetrics_{os.getpid()}.txt"
    
    def start(self):
        if os.path.exists(self.tmpfile):
            os.remove(self.tmpfile)
        self.process = subprocess.Popen(
            ["sudo", "powermetrics", "-n", "9999", "--samplers", "cpu_power", "-i", "100"],
            stdout=open(self.tmpfile, 'w'),
            stderr=subprocess.DEVNULL
        )
        time.sleep(0.5)
    
    def stop(self):
        if self.process:
            self.process.send_signal(signal.SIGINT)
            self.process.wait()
            self.process = None
        return self._parse()
    
    def _parse(self):
        powers = []
        try:
            with open(self.tmpfile) as f:
                for line in f:
                    m = re.search(r'Combined Power.*?:\s*([\d.]+)\s*mW', line)
                    if m:
                        powers.append(float(m.group(1)) / 1000.0)
        except:
            pass
        if not powers:
            return {'avg_power_w': 0, 'peak_power_w': 0, 'total_energy_j': 0, 'samples': 0}
        dt = 0.1
        energy = sum((powers[i] + powers[i+1]) / 2 * dt for i in range(len(powers)-1))
        return {
            'avg_power_w': float(np.mean(powers)),
            'peak_power_w': float(np.max(powers)),
            'total_energy_j': float(energy),
            'samples': len(powers)
        }

# --- Text prompts ---
TEXT_PROMPTS = [
    "Write a short story about a robot learning to paint.",  # Original
    "Explain the theory of general relativity in simple terms.",
    "Write a poem about the ocean at midnight.",
    "Describe the process of photosynthesis step by step.",
    "Tell me a funny joke about a programmer and a rubber duck.",
]

# --- Image prompts ---
IMAGE_PROMPTS = [
    "A serene landscape with mountains and a lake at sunset, highly detailed, 8k.",  # Original
    "A cyberpunk city street at night with neon signs and rain, photorealistic.",
    "A cute orange tabby cat sleeping on a stack of books, watercolor style.",
    "An astronaut floating in space with Earth in the background, cinematic.",
    "A medieval castle on a cliff overlooking the sea, dramatic lighting.",
]

N_REPEATS = 10
results = []

# ============ TEXT EXPERIMENT ============
print("=" * 60)
print("TEXT PROMPT DIVERSITY (Phi-3, 128 tokens, 5 prompts × 10 repeats)")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.float16, device_map="mps")
model.eval()

# Warmup
with torch.no_grad():
    inputs = tokenizer("Hello", return_tensors="pt").to("mps")
    model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.7)

pm = PowerMeasurement()

for pi, prompt in enumerate(TEXT_PROMPTS):
    print(f"\nPrompt {pi+1}/5: '{prompt[:50]}...'")
    for rep in range(N_REPEATS):
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        torch.mps.synchronize()
        
        pm.start()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7)
        torch.mps.synchronize()
        gen_time = time.time() - t0
        power = pm.stop()
        
        actual_tokens = out.shape[1] - inputs['input_ids'].shape[1]
        
        record = {
            'modality': 'text',
            'model': 'Phi-3-mini-4k',
            'prompt_id': pi,
            'prompt': prompt,
            'max_tokens': 128,
            'actual_tokens': int(actual_tokens),
            'run': rep,
            'generation_time_s': round(gen_time, 3),
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in power.items()}
        }
        results.append(record)
        print(f"  Run {rep+1}/{N_REPEATS}: {power['total_energy_j']:.1f}J, {gen_time:.2f}s, {actual_tokens} tokens")

del model, tokenizer
torch.mps.empty_cache()
import gc; gc.collect()

# ============ IMAGE EXPERIMENT ============
print("\n" + "=" * 60)
print("IMAGE PROMPT DIVERSITY (SD v1.5, 512×512, 20 steps, 5 prompts × 10 repeats)")
print("=" * 60)

from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("mps")
pipe.set_progress_bar_config(disable=True)

# Warmup
_ = pipe("warmup", num_inference_steps=2, height=256, width=256, generator=torch.Generator("mps").manual_seed(0))

for pi, prompt in enumerate(IMAGE_PROMPTS):
    print(f"\nPrompt {pi+1}/5: '{prompt[:50]}...'")
    for rep in range(N_REPEATS):
        gen = torch.Generator("mps").manual_seed(rep)
        torch.mps.synchronize()
        
        pm.start()
        t0 = time.time()
        _ = pipe(prompt, num_inference_steps=20, height=512, width=512, guidance_scale=7.5, generator=gen)
        torch.mps.synchronize()
        gen_time = time.time() - t0
        power = pm.stop()
        
        record = {
            'modality': 'image',
            'model': 'SD-v1.5',
            'prompt_id': pi,
            'prompt': prompt,
            'resolution': '512x512',
            'steps': 20,
            'run': rep,
            'generation_time_s': round(gen_time, 3),
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in power.items()}
        }
        results.append(record)
        print(f"  Run {rep+1}/{N_REPEATS}: {power['total_energy_j']:.1f}J, {gen_time:.2f}s")

# Save results
outfile = os.path.join(RESULTS_DIR, "prompt_diversity_Mac.json")
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Saved {len(results)} records to {outfile}")

# Quick analysis
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for mod in ['text', 'image']:
    mod_data = [r for r in results if r['modality'] == mod and r.get('total_energy_j', 0) > 0]
    if not mod_data:
        continue
    print(f"\n{mod.upper()}:")
    for pi in range(5):
        p_data = [r['total_energy_j'] for r in mod_data if r['prompt_id'] == pi]
        if p_data:
            print(f"  Prompt {pi+1}: {np.mean(p_data):.1f} ± {np.std(p_data):.1f}J (CV={np.std(p_data)/np.mean(p_data)*100:.1f}%)")
    
    all_energies = [r['total_energy_j'] for r in mod_data]
    # Between-prompt vs within-prompt variance
    prompt_means = [np.mean([r['total_energy_j'] for r in mod_data if r['prompt_id'] == pi]) for pi in range(5)]
    between_var = np.var(prompt_means)
    within_vars = [np.var([r['total_energy_j'] for r in mod_data if r['prompt_id'] == pi]) for pi in range(5)]
    within_var = np.mean(within_vars)
    print(f"  Between-prompt variance: {between_var:.2f}")
    print(f"  Within-prompt variance: {within_var:.2f}")
    print(f"  Ratio (between/within): {between_var/within_var:.2f}" if within_var > 0 else "")
    print(f"  Overall CV: {np.std(all_energies)/np.mean(all_energies)*100:.1f}%")
