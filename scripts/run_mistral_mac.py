#!/usr/bin/env python3
"""Run Mistral-7B experiment on Mac — temp file power monitoring."""
import torch, time, gc, json, subprocess, os, sys, signal, tempfile
import numpy as np

DEVICE = "mps"
REPEATS = 30
RESULTS_DIR = os.path.expanduser("~/Documents/project_energy/results")

class PowerMonitor:
    def __init__(self):
        self._proc = None
        self._tmpfile = None

    def start(self):
        self._tmpfile = tempfile.mktemp(prefix='pm_', suffix='.txt')
        # Run powermetrics directly (no shell=True), redirect stdout to file via python
        self._proc = subprocess.Popen(
            ['powermetrics', '-i', '100', '-n', '9999', '--samplers', 'cpu_power'],
            stdout=open(self._tmpfile, 'w'),
            stderr=subprocess.DEVNULL
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
                                if 'mW' in raw:
                                    val = float(raw.replace('mW', '').strip()) / 1000.0
                                else:
                                    val = float(raw.replace('W', '').strip())
                                readings.append(val)
                            except:
                                pass
            except:
                pass
            try: os.unlink(self._tmpfile)
            except: pass

        if len(readings) >= 2:
            powers = np.array(readings)
            dt = 0.1
            total_e = float(np.sum(powers) * dt)
            return {
                'total_energy_j': round(total_e, 2),
                'avg_power_w': round(float(np.mean(powers)), 2),
                'max_power_w': round(float(np.max(powers)), 2),
                'duration_sec': round(len(powers) * dt, 2),
                'samples': len(powers)
            }
        return {}


# Quick test
print("Testing powermetrics...", flush=True)
pm_test = PowerMonitor()
pm_test.start()
time.sleep(2)
test_result = pm_test.stop()
if 'total_energy_j' in test_result:
    print(f"  OK: {test_result['avg_power_w']:.1f} W, {test_result['samples']} samples", flush=True)
else:
    print("  FAIL: Run with sudo python3", flush=True)
    sys.exit(1)

print("Loading Mistral-7B...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')

prompt = 'Write a short story about a robot learning to paint.'
inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)

results = []
for max_tok in [64, 128, 256]:
    print(f"Mistral-7B max_tokens={max_tok}", flush=True)
    _ = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    for run in range(1, REPEATS + 1):
        torch.mps.synchronize()
        pm = PowerMonitor()
        pm.start()
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False, pad_token_id=tokenizer.eos_token_id)
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
        if run % 5 == 0:
            e = stats.get('total_energy_j', 'N/A')
            print(f"  Run {run}/{REPEATS}: {e} J ({gen_time:.1f}s)", flush=True)

outpath = os.path.join(RESULTS_DIR, 'exp6_mistral_Mac.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)
valid = sum(1 for r in results if 'total_energy_j' in r)
print(f"\nSaved {len(results)} records ({valid} with energy) to {outpath}", flush=True)
