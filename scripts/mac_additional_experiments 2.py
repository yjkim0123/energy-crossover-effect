#!/usr/bin/env python3
"""
Additional Mac M4 Pro experiments for project_energy.
Adds: more image resolutions, more video frames, SDXL, Mistral-7B
"""

import torch
import time
import json
import os
import gc
import subprocess
import re
import numpy as np
from threading import Thread, Event

RESULTS_DIR = os.path.expanduser("~/Documents/project_energy/results")
REPEATS = 30
DEVICE = "mps"

# ── Power monitoring via powermetrics ──
class PowerMonitor:
    def __init__(self):
        self.samples = []
        self._stop = Event()
        self._proc = None

    def _parse_power(self, line):
        m = re.search(r'Combined Power.*?:\s*([\d.]+)\s*mW', line)
        if m:
            return float(m.group(1)) / 1000.0
        return None

    def _record(self):
        self._proc = subprocess.Popen(
            ['sudo', 'powermetrics', '-i', '100', '--samplers', 'cpu_power,gpu_power'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            p = self._parse_power(line)
            if p is not None and p > 0:
                self.samples.append((time.time(), p))

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = Thread(target=self._record, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
        self._thread.join(timeout=5)
        if len(self.samples) < 2:
            return {}
        times = [s[0] for s in self.samples]
        powers = [s[1] for s in self.samples]
        dt = [times[i+1] - times[i] for i in range(len(times)-1)]
        energy = sum((powers[i] + powers[i+1]) / 2 * dt[i] for i in range(len(dt)))
        return {
            'total_energy_j': round(energy, 2),
            'avg_power_w': round(np.mean(powers), 2),
            'max_power_w': round(max(powers), 2),
            'duration_sec': round(times[-1] - times[0], 2),
            'samples': len(self.samples)
        }


def save_results(results, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path} ({len(results)} records)")


# ═══════════════════════════════════════════
# EXPERIMENT 1: SD v1.5 additional resolutions
# ═══════════════════════════════════════════
def run_sdv15_extra():
    print("\n" + "="*60)
    print("EXP 1: SD v1.5 additional resolutions (128², 640²)")
    print("="*60)
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        'stable-diffusion-v1-5/stable-diffusion-v1-5',
        torch_dtype=torch.float16
    ).to(DEVICE)
    prompt = 'A serene landscape with mountains and a lake at sunset, highly detailed, 8k'

    results = []
    for res in [128, 640]:
        steps = 20
        print(f"  Image {res}x{res}, {steps} steps")
        # Warmup
        _ = pipe(prompt, height=res, width=res, num_inference_steps=steps,
                guidance_scale=7.5, generator=torch.Generator(DEVICE).manual_seed(0))

        for run in range(1, REPEATS + 1):
            if DEVICE == "mps":
                torch.mps.synchronize()
            pm = PowerMonitor()
            pm.start()
            t0 = time.time()
            _ = pipe(prompt, height=res, width=res, num_inference_steps=steps,
                    guidance_scale=7.5, generator=torch.Generator(DEVICE).manual_seed(run))
            if DEVICE == "mps":
                torch.mps.synchronize()
            gen_time = time.time() - t0
            stats = pm.stop()
            stats.update({
                'generation_time_sec': round(gen_time, 2),
                'resolution': f'{res}x{res}',
                'steps': steps,
                'modality': 'image',
                'model': 'SD-v1-5',
                'params_B': 0.9,
                'run': run,
                'hardware': 'Apple M4 Pro',
                'backend': 'mps'
            })
            results.append(stats)
            if run % 10 == 0:
                print(f"    Run {run}/{REPEATS}: {stats.get('total_energy_j', 'N/A')} J")

    save_results(results, 'exp2_image_extra_Mac.json')
    del pipe
    gc.collect()
    torch.mps.empty_cache()
    return results


# ═══════════════════════════════════════════
# EXPERIMENT 2: AnimateDiff additional frames
# ═══════════════════════════════════════════
def run_animatediff_extra():
    print("\n" + "="*60)
    print("EXP 2: AnimateDiff additional frames (6, 16) at 256²")
    print("="*60)
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

    adapter = MotionAdapter.from_pretrained(
        'guoyww/animatediff-motion-adapter-v1-5-3',
        torch_dtype=torch.float16
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        'stable-diffusion-v1-5/stable-diffusion-v1-5',
        motion_adapter=adapter,
        torch_dtype=torch.float16
    ).to(DEVICE)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule='linear',
        clip_sample=False,
        timestep_spacing='linspace',
        steps_offset=1
    )
    prompt = 'A serene landscape with mountains and a lake at sunset, highly detailed, 8k'

    results = []
    for frames in [6, 16]:
        steps = 20
        print(f"  Video {frames}f, {steps} steps, 256x256")
        # Warmup
        _ = pipe(prompt, num_frames=frames, height=256, width=256,
                num_inference_steps=steps, generator=torch.Generator(DEVICE).manual_seed(0))

        for run in range(1, REPEATS + 1):
            if DEVICE == "mps":
                torch.mps.synchronize()
            pm = PowerMonitor()
            pm.start()
            t0 = time.time()
            _ = pipe(prompt, num_frames=frames, height=256, width=256,
                    num_inference_steps=steps, generator=torch.Generator(DEVICE).manual_seed(run))
            if DEVICE == "mps":
                torch.mps.synchronize()
            gen_time = time.time() - t0
            stats = pm.stop()
            stats.update({
                'generation_time_sec': round(gen_time, 2),
                'frames': frames,
                'steps': steps,
                'resolution': '256x256',
                'modality': 'video',
                'model': 'AnimateDiff-v1.5',
                'run': run,
                'hardware': 'Apple M4 Pro',
                'backend': 'mps'
            })
            results.append(stats)
            if run % 10 == 0:
                print(f"    Run {run}/{REPEATS}: {stats.get('total_energy_j', 'N/A')} J")

    save_results(results, 'exp3_video_extra_Mac.json')
    del pipe, adapter
    gc.collect()
    torch.mps.empty_cache()
    return results


# ═══════════════════════════════════════════
# EXPERIMENT 3: SDXL (second image model)
# ═══════════════════════════════════════════
def run_sdxl():
    print("\n" + "="*60)
    print("EXP 3: SDXL at 512², 768², 1024²")
    print("="*60)
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.float16,
        variant='fp16',
        use_safetensors=True
    ).to(DEVICE)
    prompt = 'A serene landscape with mountains and a lake at sunset, highly detailed, 8k'

    results = []
    for res in [512, 768, 1024]:
        steps = 20
        print(f"  SDXL {res}x{res}, {steps} steps")
        try:
            # Warmup
            _ = pipe(prompt, height=res, width=res, num_inference_steps=steps,
                    guidance_scale=7.5, generator=torch.Generator(DEVICE).manual_seed(0))
        except Exception as e:
            print(f"    SKIP {res}x{res}: {e}")
            continue

        for run in range(1, REPEATS + 1):
            if DEVICE == "mps":
                torch.mps.synchronize()
            pm = PowerMonitor()
            pm.start()
            t0 = time.time()
            _ = pipe(prompt, height=res, width=res, num_inference_steps=steps,
                    guidance_scale=7.5, generator=torch.Generator(DEVICE).manual_seed(run))
            if DEVICE == "mps":
                torch.mps.synchronize()
            gen_time = time.time() - t0
            stats = pm.stop()
            stats.update({
                'generation_time_sec': round(gen_time, 2),
                'resolution': f'{res}x{res}',
                'steps': steps,
                'modality': 'image',
                'model': 'SDXL-base-1.0',
                'params_B': 3.5,
                'run': run,
                'hardware': 'Apple M4 Pro',
                'backend': 'mps'
            })
            results.append(stats)
            if run % 10 == 0:
                print(f"    Run {run}/{REPEATS}: {stats.get('total_energy_j', 'N/A')} J")

    save_results(results, 'exp5_sdxl_Mac.json')
    del pipe
    gc.collect()
    torch.mps.empty_cache()
    return results


# ═══════════════════════════════════════════
# EXPERIMENT 4: Mistral-7B (second text model)
# ═══════════════════════════════════════════
def run_mistral():
    print("\n" + "="*60)
    print("EXP 4: Mistral-7B-Instruct at 64, 128, 256 tokens")
    print("="*60)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    print("  Loading Mistral-7B...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    prompt = 'Write a short story about a robot learning to paint.'
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

    results = []
    for max_tok in [64, 128, 256]:
        print(f"  Mistral-7B max_tokens={max_tok}")
        # Warmup
        _ = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)

        for run in range(1, REPEATS + 1):
            if DEVICE == "mps":
                torch.mps.synchronize()
            pm = PowerMonitor()
            pm.start()
            t0 = time.time()
            out = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)
            if DEVICE == "mps":
                torch.mps.synchronize()
            gen_time = time.time() - t0
            stats = pm.stop()
            stats.update({
                'generation_time_sec': round(gen_time, 2),
                'max_tokens': max_tok,
                'modality': 'text',
                'model': 'Mistral-7B-Instruct',
                'params_B': 7.2,
                'run': run,
                'hardware': 'Apple M4 Pro',
                'backend': 'mps'
            })
            results.append(stats)
            if run % 10 == 0:
                print(f"    Run {run}/{REPEATS}: {stats.get('total_energy_j', 'N/A')} J")

    save_results(results, 'exp6_mistral_Mac.json')
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return results


# ═══════════════════════════════════════════
if __name__ == '__main__':
    import datetime
    print(f"Starting additional experiments at {datetime.datetime.now()}")
    print(f"Device: {DEVICE}, Repeats: {REPEATS}")

    all_results = {}

    # Run in order of memory usage (smallest first)
    all_results['sdv15_extra'] = run_sdv15_extra()
    all_results['animatediff_extra'] = run_animatediff_extra()
    all_results['sdxl'] = run_sdxl()
    all_results['mistral'] = run_mistral()

    print(f"\n{'='*60}")
    print(f"ALL DONE at {datetime.datetime.now()}")
    total = sum(len(v) for v in all_results.values())
    print(f"Total additional runs: {total}")
    for k, v in all_results.items():
        print(f"  {k}: {len(v)} runs")
