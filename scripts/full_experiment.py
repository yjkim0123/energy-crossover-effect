#!/usr/bin/env python3
"""
Full Cross-Modal Energy Experiment for Nature Communications.
Runs all 4 modalities × multiple configs × 30 repeats.
Estimated runtime: ~5 hours on Mac mini M4 Pro.
"""
import os, time, json, subprocess, threading, gc, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np

RESULTS_DIR = "/Users/yongjun_kim/Documents/project_energy/results"
REPEATS = 30
device = "mps"

# ============================================================
# Power Monitor
# ============================================================
class PowerMonitor:
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
        self._thread.join(timeout=3)
        if not self.readings or len(self.readings) < 2:
            return {"total_energy_j": 0, "avg_power_w": 0, "max_power_w": 0, "duration_sec": 0, "samples": 0}
        powers = [r[1] for r in self.readings]
        duration = self.readings[-1][0] - self.readings[0][0]
        energy_j = sum((self.readings[i][1]+self.readings[i-1][1])/2 *
                       (self.readings[i][0]-self.readings[i-1][0])
                       for i in range(1, len(self.readings)))
        return {"total_energy_j": round(energy_j,2), "avg_power_w": round(sum(powers)/len(powers),2),
                "max_power_w": round(max(powers),2), "duration_sec": round(duration,2), "samples": len(self.readings)}

def run_and_measure(task_fn, pm, cooldown=3):
    """Run task with power measurement."""
    time.sleep(cooldown)
    pm.start()
    time.sleep(0.5)
    t0 = time.time()
    extra = task_fn()
    gen_time = time.time() - t0
    time.sleep(0.5)
    result = pm.stop()
    result["generation_time_sec"] = round(gen_time, 2)
    if extra:
        result.update(extra)
    return result

def save_results(data, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  💾 Saved {filename} ({len(data)} records)")

def print_summary(results, key="total_energy_j"):
    vals = [r[key] for r in results if r.get(key, 0) > 0]
    if vals:
        print(f"  📊 {key}: {np.mean(vals):.1f} ± {np.std(vals):.1f} (n={len(vals)})")

# ============================================================
# EXPERIMENT 1: TEXT
# ============================================================
def run_text_experiments():
    print("\n" + "="*60)
    print("📝 EXPERIMENT 1: TEXT GENERATION")
    print("="*60)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.float16, device_map=device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: Phi-3-mini ({params/1e9:.1f}B) on {device}")
    
    # Warm up
    inp = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad(): model.generate(**inp, max_new_tokens=16)
    
    prompt = "Explain the impact of artificial intelligence on renewable energy systems in detail."
    token_configs = [64, 128, 256, 512]
    all_results = []
    pm = PowerMonitor()
    
    for max_tok in token_configs:
        print(f"\n  📏 {max_tok} tokens × {REPEATS} runs")
        for run in range(REPEATS):
            def task():
                inp = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=max_tok, do_sample=True, temperature=0.7)
                n_tok = out.shape[1] - inp["input_ids"].shape[1]
                return {"tokens_generated": int(n_tok)}
            
            result = run_and_measure(task, pm, cooldown=2)
            result["modality"] = "text"
            result["model"] = "Phi-3-mini-4k"
            result["params_B"] = round(params/1e9, 1)
            result["max_tokens"] = max_tok
            result["run"] = run + 1
            all_results.append(result)
            
            if (run+1) % 10 == 0 or run == 0:
                print(f"    Run {run+1}/{REPEATS}: {result['total_energy_j']:.0f}J | {result['avg_power_w']:.1f}W | {result['generation_time_sec']:.1f}s")
        
        print_summary([r for r in all_results if r["max_tokens"]==max_tok])
    
    save_results(all_results, "exp1_text.json")
    del model, tokenizer; gc.collect()
    if hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
    return all_results

# ============================================================
# EXPERIMENT 2: IMAGE
# ============================================================
def run_image_experiments():
    print("\n" + "="*60)
    print("🖼️  EXPERIMENT 2: IMAGE GENERATION")
    print("="*60)
    
    from diffusers import StableDiffusionPipeline
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to(device)
    params = sum(p.numel() for p in pipe.unet.parameters())
    print(f"  Model: SD v1.5 (UNet {params/1e9:.1f}B) on {device}")
    
    # Warm up
    with torch.no_grad(): pipe("test", num_inference_steps=3, height=256, width=256)
    
    prompt = "A beautiful sunset over mountains with a river, photorealistic"
    configs = [
        (256, 20), (256, 30),
        (384, 20), (384, 30),
        (512, 20), (512, 30),
    ]
    all_results = []
    pm = PowerMonitor()
    
    for res, steps in configs:
        print(f"\n  📏 {res}x{res}, {steps} steps × {REPEATS} runs")
        for run in range(REPEATS):
            def task(r=res, s=steps):
                with torch.no_grad():
                    pipe(prompt, num_inference_steps=s, height=r, width=r)
                return {"resolution": f"{r}x{r}", "steps": s}
            
            result = run_and_measure(task, pm, cooldown=2)
            result["modality"] = "image"
            result["model"] = "SD-v1-5"
            result["params_B"] = round(params/1e9, 1)
            result["resolution"] = f"{res}x{res}"
            result["steps"] = steps
            result["run"] = run + 1
            all_results.append(result)
            
            if (run+1) % 10 == 0 or run == 0:
                print(f"    Run {run+1}/{REPEATS}: {result['total_energy_j']:.0f}J | {result['avg_power_w']:.1f}W | {result['generation_time_sec']:.1f}s")
        
        print_summary([r for r in all_results if r["resolution"]==f"{res}x{res}" and r["steps"]==steps])
    
    save_results(all_results, "exp2_image.json")
    del pipe; gc.collect()
    if hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
    return all_results

# ============================================================
# EXPERIMENT 3: VIDEO
# ============================================================
def run_video_experiments():
    print("\n" + "="*60)
    print("🎬 EXPERIMENT 3: VIDEO GENERATION")
    print("="*60)
    
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
    pipe = pipe.to(device)
    print(f"  Model: AnimateDiff on {device}")
    
    # Warm up
    with torch.no_grad(): pipe("test", num_frames=4, num_inference_steps=3, height=256, width=256)
    
    prompt = "A beautiful sunset over mountains, timelapse, cinematic"
    # 256x256 only (512 OOMs)
    configs = [
        (4, 20), (8, 20), (12, 20),
        (4, 30), (8, 30),
    ]
    all_results = []
    pm = PowerMonitor()
    
    for frames, steps in configs:
        print(f"\n  📏 {frames} frames, 256x256, {steps} steps × {REPEATS} runs")
        for run in range(REPEATS):
            def task(f=frames, s=steps):
                with torch.no_grad():
                    out = pipe(prompt, num_frames=f, num_inference_steps=s, height=256, width=256)
                return {"frames": f, "steps": s}
            
            result = run_and_measure(task, pm, cooldown=3)
            result["modality"] = "video"
            result["model"] = "AnimateDiff-v1.5"
            result["resolution"] = "256x256"
            result["frames"] = frames
            result["steps"] = steps
            result["run"] = run + 1
            all_results.append(result)
            
            if (run+1) % 10 == 0 or run == 0:
                print(f"    Run {run+1}/{REPEATS}: {result['total_energy_j']:.0f}J | {result['avg_power_w']:.1f}W | {result['generation_time_sec']:.1f}s")
        
        print_summary([r for r in all_results if r["frames"]==frames and r["steps"]==steps])
    
    save_results(all_results, "exp3_video.json")
    del pipe, adapter; gc.collect()
    if hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
    return all_results

# ============================================================
# EXPERIMENT 4: MUSIC
# ============================================================
def run_music_experiments():
    print("\n" + "="*60)
    print("🎵 EXPERIMENT 4: MUSIC GENERATION")
    print("="*60)
    
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    
    prompt = "upbeat electronic dance music with a catchy melody"
    token_configs = [128, 256, 512]
    model_configs = [
        ("facebook/musicgen-small", "MusicGen-small"),
    ]
    all_results = []
    pm = PowerMonitor()
    
    for model_name, model_label in model_configs:
        print(f"\n  🎵 Loading {model_label}...")
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"     Params: {params/1e6:.0f}M")
        
        # Warm up
        inp = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        with torch.no_grad(): model.generate(**inp, max_new_tokens=32)
        
        for max_tok in token_configs:
            print(f"\n  📏 {model_label}, {max_tok} tokens × {REPEATS} runs")
            for run in range(REPEATS):
                def task(mt=max_tok):
                    inp = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
                    with torch.no_grad():
                        audio = model.generate(**inp, max_new_tokens=mt)
                    audio_len = audio.shape[-1] / model.config.audio_encoder.sampling_rate
                    return {"audio_sec": round(audio_len, 1), "max_tokens": mt}
                
                result = run_and_measure(task, pm, cooldown=2)
                result["modality"] = "music"
                result["model"] = model_label
                result["params_M"] = round(params/1e6, 0)
                result["max_tokens"] = max_tok
                result["run"] = run + 1
                all_results.append(result)
                
                if (run+1) % 10 == 0 or run == 0:
                    print(f"    Run {run+1}/{REPEATS}: {result['total_energy_j']:.0f}J | {result['avg_power_w']:.1f}W | {result['generation_time_sec']:.1f}s")
            
            print_summary([r for r in all_results if r["model"]==model_label and r["max_tokens"]==max_tok])
        
        del model, processor; gc.collect()
        if hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
    
    # Medium (128 and 256 tokens only)
    print(f"\n  🎵 Loading MusicGen-medium...")
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium").to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"     Params: {params/1e6:.0f}M")
    
    inp = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    with torch.no_grad(): model.generate(**inp, max_new_tokens=32)
    
    for max_tok in [128, 256]:
        print(f"\n  📏 MusicGen-medium, {max_tok} tokens × {REPEATS} runs")
        for run in range(REPEATS):
            def task(mt=max_tok):
                inp = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    audio = model.generate(**inp, max_new_tokens=mt)
                audio_len = audio.shape[-1] / model.config.audio_encoder.sampling_rate
                return {"audio_sec": round(audio_len, 1), "max_tokens": mt}
            
            result = run_and_measure(task, pm, cooldown=2)
            result["modality"] = "music"
            result["model"] = "MusicGen-medium"
            result["params_M"] = round(params/1e6, 0)
            result["max_tokens"] = max_tok
            result["run"] = run + 1
            all_results.append(result)
            
            if (run+1) % 10 == 0 or run == 0:
                print(f"    Run {run+1}/{REPEATS}: {result['total_energy_j']:.0f}J | {result['avg_power_w']:.1f}W | {result['generation_time_sec']:.1f}s")
        
        print_summary([r for r in all_results if r["model"]=="MusicGen-medium" and r["max_tokens"]==max_tok])
    
    save_results(all_results, "exp4_music.json")
    del model, processor; gc.collect()
    return all_results

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start_time = time.time()
    print("🚀 FULL CROSS-MODAL ENERGY EXPERIMENT")
    print(f"   Device: {device}")
    print(f"   Repeats: {REPEATS}")
    print(f"   Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_data = {}
    
    try:
        all_data["text"] = run_text_experiments()
    except Exception as e:
        print(f"❌ Text failed: {e}")
    
    try:
        all_data["image"] = run_image_experiments()
    except Exception as e:
        print(f"❌ Image failed: {e}")
    
    try:
        all_data["video"] = run_video_experiments()
    except Exception as e:
        print(f"❌ Video failed: {e}")
    
    try:
        all_data["music"] = run_music_experiments()
    except Exception as e:
        print(f"❌ Music failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ ALL EXPERIMENTS COMPLETE!")
    print(f"   Total time: {total_time/3600:.1f} hours")
    print(f"   End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Results in: {RESULTS_DIR}/")
    print(f"{'='*60}")
