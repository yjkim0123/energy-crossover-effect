#!/usr/bin/env python3
"""H100 Full Energy-Quality Benchmark — runs all experiments in one go."""
import torch, time, json, os, gc, datetime
import numpy as np
import pynvml
from threading import Thread, Event

assert torch.cuda.is_available(), 'No GPU!'

SAVE_DIR = "/content/project_energy_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
GPU_NAME = pynvml.nvmlDeviceGetName(handle)
if isinstance(GPU_NAME, bytes): GPU_NAME = GPU_NAME.decode()
gpu_tag = GPU_NAME.replace(' ','-').replace('/','-')
REPEATS = 30

print(f'GPU: {GPU_NAME}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'Save: {SAVE_DIR}')

class PowerMonitor:
    def __init__(self):
        self.samples = []
        self._stop = Event()
    def _record(self):
        while not self._stop.is_set():
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                self.samples.append((time.time(), p))
            except: pass
            time.sleep(0.05)
    def start(self):
        self.samples = []; self._stop.clear()
        self._thread = Thread(target=self._record, daemon=True)
        self._thread.start()
    def stop(self):
        self._stop.set(); self._thread.join()
        if len(self.samples) < 2: return {}
        times, powers = zip(*self.samples)
        dt = [times[i+1]-times[i] for i in range(len(times)-1)]
        energy = sum((powers[i]+powers[i+1])/2*dt[i] for i in range(len(dt)))
        return {
            'total_energy_j': round(energy,2),
            'avg_power_w': round(np.mean(powers),2),
            'max_power_w': round(max(powers),2),
            'duration_sec': round(times[-1]-times[0],2),
            'samples': len(self.samples)
        }

def save(data, name):
    path = f'{SAVE_DIR}/{name}_{gpu_tag}.json'
    with open(path, 'w') as f: json.dump(data, f, indent=2)
    print(f'  Saved {len(data)} records to {path}')

print('Setup OK')

# ========== CLIP helper ==========
import open_clip
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None

def get_clip_score(image, text):
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        _clip_model = _clip_model.to('cuda').eval()
        _clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    img_input = _clip_preprocess(image).unsqueeze(0).to('cuda')
    txt_input = _clip_tokenizer([text]).to('cuda')
    with torch.no_grad():
        img_feat = _clip_model.encode_image(img_input)
        txt_feat = _clip_model.encode_text(txt_input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        score = (img_feat @ txt_feat.T).item()
    return round(score, 4)

prompt_text = 'Write a short story about a robot learning to paint.'
prompt_img = 'A serene landscape with mountains and a lake at sunset, highly detailed, 8k'

# ========== 1. TEXT: Phi-3 ==========
print('\n=== TEXT: Phi-3 ===')
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'microsoft/Phi-3-mini-4k-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='cuda')
inputs = tokenizer(prompt_text, return_tensors='pt').to('cuda')

results = []
for max_tok in [32, 64, 128, 256, 512]:
    print(f'  Phi-3 {max_tok} tok')
    _ = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)
    for run in range(1, REPEATS+1):
        torch.cuda.synchronize()
        pm = PowerMonitor(); pm.start()
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)
        torch.cuda.synchronize()
        gen_time = time.time() - t0
        stats = pm.stop()
        stats.update({
            'generation_time_sec': round(gen_time, 2),
            'max_tokens': max_tok,
            'actual_tokens': len(out[0]) - inputs['input_ids'].shape[1],
            'modality': 'text', 'model': 'Phi-3-mini-4k', 'params_B': 3.8,
            'run': run, 'hardware': GPU_NAME, 'backend': 'cuda'
        })
        results.append(stats)
save(results, 'text_phi3')
del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
print('Phi-3 done')

# ========== 2. IMAGE: SD v1.5 + CLIP ==========
print('\n=== IMAGE: SD v1.5 ===')
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    'stable-diffusion-v1-5/stable-diffusion-v1-5', torch_dtype=torch.float16
).to('cuda')

results = []
for res in [128, 256, 384, 512, 640]:
    for steps in [5, 10, 15, 20, 30, 50]:
        print(f'  SD v1.5 {res}x{res} {steps}steps')
        _ = pipe(prompt_img, height=res, width=res, num_inference_steps=steps,
                guidance_scale=7.5, generator=torch.Generator('cuda').manual_seed(0))
        for run in range(1, REPEATS+1):
            torch.cuda.synchronize()
            pm = PowerMonitor(); pm.start()
            t0 = time.time()
            output = pipe(prompt_img, height=res, width=res, num_inference_steps=steps,
                    guidance_scale=7.5, generator=torch.Generator('cuda').manual_seed(run))
            torch.cuda.synchronize()
            gen_time = time.time() - t0
            stats = pm.stop()
            clip_s = get_clip_score(output.images[0], prompt_img) if run <= 3 else 0
            stats.update({
                'generation_time_sec': round(gen_time, 2),
                'resolution': f'{res}x{res}', 'steps': steps, 'clip_score': clip_s,
                'modality': 'image', 'model': 'SD-v1-5', 'params_B': 0.9,
                'run': run, 'hardware': GPU_NAME, 'backend': 'cuda'
            })
            results.append(stats)
        save(results, 'image_sd15')
del pipe; gc.collect(); torch.cuda.empty_cache()
print('SD v1.5 done')

# ========== 3. IMAGE: SDXL + CLIP ==========
print('\n=== IMAGE: SDXL ===')
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16, variant='fp16'
).to('cuda')

results = []
for res in [512, 640, 768, 1024]:
    for steps in [5, 10, 15, 20, 30]:
        print(f'  SDXL {res}x{res} {steps}steps')
        try:
            _ = pipe(prompt_img, height=res, width=res, num_inference_steps=steps,
                    generator=torch.Generator('cuda').manual_seed(0))
        except Exception as e:
            print(f'    SKIP: {e}')
            continue
        for run in range(1, REPEATS+1):
            torch.cuda.synchronize()
            pm = PowerMonitor(); pm.start()
            t0 = time.time()
            output = pipe(prompt_img, height=res, width=res, num_inference_steps=steps,
                    generator=torch.Generator('cuda').manual_seed(run))
            torch.cuda.synchronize()
            gen_time = time.time() - t0
            stats = pm.stop()
            clip_s = get_clip_score(output.images[0], prompt_img) if run <= 3 else 0
            stats.update({
                'generation_time_sec': round(gen_time, 2),
                'resolution': f'{res}x{res}', 'steps': steps, 'clip_score': clip_s,
                'modality': 'image', 'model': 'SDXL-base-1.0', 'params_B': 3.5,
                'run': run, 'hardware': GPU_NAME, 'backend': 'cuda'
            })
            results.append(stats)
        save(results, 'image_sdxl')
del pipe; gc.collect(); torch.cuda.empty_cache()
print('SDXL done')

# ========== 4. VIDEO: AnimateDiff ==========
print('\n=== VIDEO: AnimateDiff ===')
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler

adapter = MotionAdapter.from_pretrained(
    'guoyww/animatediff-motion-adapter-v1-5-3', torch_dtype=torch.float16)
vpipe = AnimateDiffPipeline.from_pretrained(
    'stable-diffusion-v1-5/stable-diffusion-v1-5',
    motion_adapter=adapter, torch_dtype=torch.float16
).to('cuda')
vpipe.scheduler = DDIMScheduler.from_config(
    vpipe.scheduler.config, beta_schedule='linear',
    clip_sample=False, timestep_spacing='linspace', steps_offset=1)

results = []
for frames in [4, 6, 8, 12, 16]:
    for steps in [10, 20, 30]:
        print(f'  AnimateDiff {frames}f {steps}steps')
        try:
            _ = vpipe(prompt_img, num_frames=frames, height=256, width=256,
                     num_inference_steps=steps, generator=torch.Generator('cuda').manual_seed(0))
        except Exception as e:
            print(f'    SKIP: {e}')
            continue
        for run in range(1, REPEATS+1):
            torch.cuda.synchronize()
            pm = PowerMonitor(); pm.start()
            t0 = time.time()
            _ = vpipe(prompt_img, num_frames=frames, height=256, width=256,
                     num_inference_steps=steps, generator=torch.Generator('cuda').manual_seed(run))
            torch.cuda.synchronize()
            gen_time = time.time() - t0
            stats = pm.stop()
            stats.update({
                'generation_time_sec': round(gen_time, 2),
                'frames': frames, 'steps': steps, 'resolution': '256x256',
                'modality': 'video', 'model': 'AnimateDiff-v1.5',
                'run': run, 'hardware': GPU_NAME, 'backend': 'cuda'
            })
            results.append(stats)
        save(results, 'video_animatediff')
del vpipe, adapter; gc.collect(); torch.cuda.empty_cache()
print('AnimateDiff done')

# ========== 5. BATCHED: Phi-3 ==========
print('\n=== BATCHED: Phi-3 ===')
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Phi-3-mini-4k-instruct', torch_dtype=torch.float16, device_map='cuda')

results = []
for bs in [1, 2, 4, 8, 16]:
    batch = [prompt_text] * bs
    inp = tokenizer(batch, return_tensors='pt', padding=True).to('cuda')
    print(f'  Batch={bs}')
    _ = model.generate(**inp, max_new_tokens=256, do_sample=False)
    for run in range(1, REPEATS+1):
        torch.cuda.synchronize()
        pm = PowerMonitor(); pm.start()
        t0 = time.time()
        _ = model.generate(**inp, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize()
        gen_time = time.time() - t0
        stats = pm.stop()
        pqe = stats.get('total_energy_j', 0) / bs
        stats.update({
            'generation_time_sec': round(gen_time, 2),
            'max_tokens': 256, 'batch_size': bs,
            'per_query_energy_j': round(pqe, 2),
            'modality': 'text_batched', 'model': 'Phi-3-mini-4k', 'params_B': 3.8,
            'run': run, 'hardware': GPU_NAME, 'backend': 'cuda'
        })
        results.append(stats)
save(results, 'batched_phi3')
del model, tokenizer; gc.collect(); torch.cuda.empty_cache()
print('Batched done')

# ========== SUMMARY ==========
print(f'\n===== ALL DONE on {GPU_NAME}! =====')
print(f'Results in: {SAVE_DIR}')
for f in sorted(os.listdir(SAVE_DIR)):
    if f.endswith('.json') and gpu_tag in f:
        with open(os.path.join(SAVE_DIR, f)) as fh:
            d = json.load(fh)
        print(f'  {f}: {len(d)} records')

# ========== DUMP ALL JSON TO STDOUT ==========
print('\n===== JSON_DUMP_START =====')
import base64, gzip
for f in sorted(os.listdir(SAVE_DIR)):
    if f.endswith('.json'):
        with open(os.path.join(SAVE_DIR, f), 'rb') as fh:
            raw = fh.read()
        compressed = gzip.compress(raw)
        b64 = base64.b64encode(compressed).decode()
        print(f'JSONFILE:{f}:{b64}')
print('===== JSON_DUMP_END =====')
