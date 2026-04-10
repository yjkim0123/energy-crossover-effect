#!/usr/bin/env python3
"""Fix figures: colorblind-safe palette, clean labels, fix CLIP Pareto plot."""
import json, os, sys
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

# Colorblind-safe palette
MAC_COLOR = '#0072B2'   # blue
A100_COLOR = '#D55E00'  # vermillion/orange  
H100_COLOR = '#009E73'  # bluish green
MAC_MARKER = 'o'
A100_MARKER = 's'
H100_MARKER = '^'

RESULTS = os.path.expanduser('~/Documents/project_energy/results')
FIGURES = os.path.expanduser('~/Documents/project_energy/figures')

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def load_json(fname):
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

def group_by(data, key_fn):
    groups = defaultdict(list)
    for r in data:
        if 'total_energy_j' not in r:
            continue  # skip incomplete records
        k = key_fn(r)
        if k is not None:
            groups[k].append(r)
    return groups

def mean_std(vals):
    a = np.array(vals)
    return a.mean(), a.std()

# ============================================================
# Load all data
# ============================================================

# Text
mac_text = load_json('exp1_text.json')
a100_text = load_json('exp1_text_A100.json')
h100_text = load_json('text_phi3_H100.json')

# Image SD v1.5
mac_img = load_json('exp2_image.json') + load_json('exp2_image_extra_Mac.json')
a100_img = load_json('exp2_image_A100.json') + load_json('exp2_image_extra_A100.json')
h100_img = load_json('image_sd15_H100.json')

# Video
mac_vid = load_json('exp3_video.json') + load_json('exp3_video_extra_Mac.json')
a100_vid = load_json('exp3_video_A100.json') + load_json('exp3_video_extra_A100.json')
h100_vid = load_json('video_animatediff_H100.json')

# Music
mac_mus = load_json('exp4_music.json')
a100_mus = load_json('exp4_music_A100.json')

# SDXL
mac_sdxl = load_json('exp5_sdxl_Mac.json')
a100_sdxl = load_json('exp5_sdxl_A100.json')
h100_sdxl = load_json('image_sdxl_H100.json')

# Batched
a100_batch = load_json('exp7_batched_A100.json')
h100_batch = load_json('batched_phi3_H100.json')

# Quality
mac_quality = load_json('quality_image_sd15_Mac.json')

# ============================================================
# Helper: get token/resolution key
# ============================================================
def get_tokens(r):
    return r.get('max_tokens') or r.get('actual_tokens') or r.get('max_new_tokens')

def get_res_steps(r):
    res = r.get('resolution', '')
    steps = r.get('steps', r.get('num_inference_steps', 20))
    return (res, steps)

def res_to_pixels(res_str):
    if 'x' in str(res_str):
        parts = str(res_str).split('x')
        return int(parts[0]) * int(parts[1])
    return 0

def get_frames_steps(r):
    frames = r.get('frames', r.get('num_frames', 0))
    steps = r.get('steps', r.get('num_inference_steps', 20))
    return (frames, steps)

# ============================================================
# FIG 1: 4-panel comparison (colorblind-safe)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Text
ax = axes[0, 0]
for data, color, marker, label in [
    (mac_text, MAC_COLOR, MAC_MARKER, 'Mac M4 Pro'),
    (a100_text, A100_COLOR, A100_MARKER, 'A100'),
    (h100_text, H100_COLOR, H100_MARKER, 'H100')
]:
    groups = group_by(data, get_tokens)
    tokens_sorted = sorted(groups.keys())
    means = [mean_std([r['total_energy_j'] for r in groups[t]])[0] for t in tokens_sorted]
    stds = [mean_std([r['total_energy_j'] for r in groups[t]])[1] for t in tokens_sorted]
    ax.errorbar(tokens_sorted, means, yerr=stds, marker=marker, color=color, 
                label=label, capsize=3, linewidth=1.5, markersize=6)
ax.set_xlabel('Max Tokens')
ax.set_ylabel('Energy (J)')
ax.set_yscale('log')
ax.set_title('(a) Text Generation (Phi-3)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# (b) Image SD v1.5 (20 steps only for clean comparison)
ax = axes[0, 1]
for data, color, marker, label in [
    (mac_img, MAC_COLOR, MAC_MARKER, 'Mac M4 Pro'),
    (a100_img, A100_COLOR, A100_MARKER, 'A100'),
    (h100_img, H100_COLOR, H100_MARKER, 'H100')
]:
    groups = group_by(data, get_res_steps)
    # Filter to steps=20
    pixels_energy = []
    for (res, steps), runs in groups.items():
        if steps == 20:
            px = res_to_pixels(res)
            if px > 0:
                m, s = mean_std([r['total_energy_j'] for r in runs])
                pixels_energy.append((px, m, s))
    pixels_energy.sort()
    if pixels_energy:
        xs = [p[0] for p in pixels_energy]
        ys = [p[1] for p in pixels_energy]
        es = [p[2] for p in pixels_energy]
        ax.errorbar(xs, ys, yerr=es, marker=marker, color=color,
                    label=label, capsize=3, linewidth=1.5, markersize=6)
ax.set_xlabel('Pixels (width × height)')
ax.set_ylabel('Energy (J)')
ax.set_title('(b) Image Generation (SD v1.5, 20 steps)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
# Add crossover annotation
ax.axhline(y=0, color='gray', alpha=0)

# (c) Video (20 steps)
ax = axes[1, 0]
for data, color, marker, label in [
    (mac_vid, MAC_COLOR, MAC_MARKER, 'Mac M4 Pro'),
    (a100_vid, A100_COLOR, A100_MARKER, 'A100'),
    (h100_vid, H100_COLOR, H100_MARKER, 'H100')
]:
    groups = group_by(data, get_frames_steps)
    frame_energy = []
    for (frames, steps), runs in groups.items():
        if steps == 20 and frames > 0:
            m, s = mean_std([r['total_energy_j'] for r in runs])
            frame_energy.append((frames, m, s))
    frame_energy.sort()
    if frame_energy:
        xs = [p[0] for p in frame_energy]
        ys = [p[1] for p in frame_energy]
        es = [p[2] for p in frame_energy]
        ax.errorbar(xs, ys, yerr=es, marker=marker, color=color,
                    label=label, capsize=3, linewidth=1.5, markersize=6)
ax.set_xlabel('Number of Frames')
ax.set_ylabel('Energy (J)')
ax.set_title('(c) Video Generation (AnimateDiff, 20 steps)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# (d) Music
ax = axes[1, 1]
for data, color, marker, label in [
    (mac_mus, MAC_COLOR, MAC_MARKER, 'Mac M4 Pro'),
    (a100_mus, A100_COLOR, A100_MARKER, 'A100'),
]:
    # Group by model + tokens
    def music_key(r):
        model = r.get('model', 'unknown')
        tokens = get_tokens(r)
        return (model, tokens)
    groups = group_by(data, music_key)
    # Plot small model
    small_items = [(t, runs) for (model, t), runs in groups.items() if 'small' in str(model).lower()]
    small_items.sort()
    if small_items:
        xs = [t for t, _ in small_items]
        ys = [mean_std([r['total_energy_j'] for r in runs])[0] for _, runs in small_items]
        es = [mean_std([r['total_energy_j'] for r in runs])[1] for _, runs in small_items]
        ax.errorbar(xs, ys, yerr=es, marker=marker, color=color,
                    label=label, capsize=3, linewidth=1.5, markersize=6)
ax.set_xlabel('Max Tokens')
ax.set_ylabel('Energy (J)')
ax.set_yscale('log')
ax.set_title('(d) Music Generation (MusicGen-small)')
ax.annotate('No H100 data\n(music not tested)', xy=(0.6, 0.4), xycoords='axes fraction',
            fontsize=10, color='gray', style='italic', ha='center')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'fig1_3hw_comparison.png'))
plt.savefig(os.path.join(FIGURES, 'fig1_3hw_comparison.pdf'))
plt.close()
print("✅ fig1_3hw_comparison saved")

# ============================================================
# FIG 2: Efficiency ratio (colorblind-safe)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def plot_ratio_panel(ax, mac_data, a100_data, h100_data, key_fn, xlabel, title):
    mac_groups = group_by(mac_data, key_fn)
    a100_groups = group_by(a100_data, key_fn)
    h100_groups = group_by(h100_data, key_fn) if h100_data else {}
    
    common_keys_a100 = sorted(set(mac_groups.keys()) & set(a100_groups.keys()))
    common_keys_h100 = sorted(set(mac_groups.keys()) & set(h100_groups.keys()))
    
    if common_keys_a100:
        xs = list(range(len(common_keys_a100)))
        ratios = []
        for k in common_keys_a100:
            mac_e = np.mean([r['total_energy_j'] for r in mac_groups[k]])
            a100_e = np.mean([r['total_energy_j'] for r in a100_groups[k]])
            ratios.append(a100_e / mac_e)
        ax.plot(xs, ratios, marker=A100_MARKER, color=A100_COLOR, label='A100/Mac', linewidth=1.5, markersize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(k) for k in common_keys_a100], rotation=45, ha='right', fontsize=9)
    
    if common_keys_h100:
        xs = list(range(len(common_keys_h100)))
        ratios = []
        for k in common_keys_h100:
            mac_e = np.mean([r['total_energy_j'] for r in mac_groups[k]])
            h100_e = np.mean([r['total_energy_j'] for r in h100_groups[k]])
            ratios.append(h100_e / mac_e)
        ax.plot(xs[:len(ratios)], ratios, marker=H100_MARKER, color=H100_COLOR, label='H100/Mac', linewidth=1.5, markersize=7)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.fill_between(ax.get_xlim(), 0, 1, alpha=0.1, color='green', label='GPU more efficient')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Energy Ratio (GPU / Mac)')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

# Text
plot_ratio_panel(axes[0, 0], mac_text, a100_text, h100_text, get_tokens,
                 'Max Tokens', '(a) Text (Phi-3)')

# Image (20 steps)
def img20_key(r):
    steps = r.get('steps', r.get('num_inference_steps', 0))
    if steps == 20:
        return r.get('resolution', '')
    return None
plot_ratio_panel(axes[0, 1], mac_img, a100_img, h100_img, img20_key,
                 'Resolution', '(b) Image (SD v1.5, 20 steps)')

# Video (20 steps)  
def vid20_key(r):
    steps = r.get('steps', r.get('num_inference_steps', 0))
    if steps == 20:
        return r.get('frames', r.get('num_frames', 0))
    return None
plot_ratio_panel(axes[1, 0], mac_vid, a100_vid, h100_vid, vid20_key,
                 'Frames', '(c) Video (AnimateDiff, 20 steps)')

# Music
def music_small_key(r):
    model = r.get('model', '')
    if 'small' in str(model).lower():
        return get_tokens(r)
    return None
plot_ratio_panel(axes[1, 1], mac_mus, a100_mus, [], music_small_key,
                 'Max Tokens', '(d) Music (MusicGen-small)')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'fig2_efficiency_ratio_3hw.png'))
plt.savefig(os.path.join(FIGURES, 'fig2_efficiency_ratio_3hw.pdf'))
plt.close()
print("✅ fig2_efficiency_ratio_3hw saved")

# ============================================================
# FIG 7: Energy-Quality Pareto (FIXED: clean labels, proper CLIP)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

# Mac quality data
mac_q_groups = group_by(mac_quality, get_res_steps)
mac_points = []
for (res, steps), runs in mac_q_groups.items():
    clips = [r['clip_score'] for r in runs if r.get('clip_score', 0) > 0]
    energies = [r['total_energy_j'] for r in runs]
    if clips and energies:
        mac_points.append({
            'energy': np.mean(energies),
            'clip': np.mean(clips),
            'res': res, 'steps': steps
        })

# H100 SD v1.5 quality data  
h100_q_groups = group_by(h100_img, get_res_steps)
h100_sd_points = []
for (res, steps), runs in h100_q_groups.items():
    clips = [r['clip_score'] for r in runs if r.get('clip_score', 0) > 0.05]  # filter zeros/noise
    energies = [r['total_energy_j'] for r in runs]
    if clips and energies:
        h100_sd_points.append({
            'energy': np.mean(energies),
            'clip': np.mean(clips),
            'res': res, 'steps': steps
        })

# H100 SDXL quality data
h100_sdxl_groups = group_by(h100_sdxl, get_res_steps)
h100_sdxl_points = []
for (res, steps), runs in h100_sdxl_groups.items():
    clips = [r['clip_score'] for r in runs if r.get('clip_score', 0) > 0.05]
    energies = [r['total_energy_j'] for r in runs]
    if clips and energies:
        h100_sdxl_points.append({
            'energy': np.mean(energies),
            'clip': np.mean(clips),
            'res': res, 'steps': steps
        })

# Plot Mac
if mac_points:
    ax.scatter([p['energy'] for p in mac_points], [p['clip'] for p in mac_points],
               c=MAC_COLOR, marker=MAC_MARKER, s=80, label='Mac SD v1.5', alpha=0.8, edgecolors='white', linewidth=0.5)

# Plot H100 SD v1.5
if h100_sd_points:
    ax.scatter([p['energy'] for p in h100_sd_points], [p['clip'] for p in h100_sd_points],
               c=H100_COLOR, marker=H100_MARKER, s=80, label='H100 SD v1.5', alpha=0.8, edgecolors='white', linewidth=0.5)

# Plot H100 SDXL
if h100_sdxl_points:
    ax.scatter([p['energy'] for p in h100_sdxl_points], [p['clip'] for p in h100_sdxl_points],
               c='#CC79A7', marker='D', s=80, label='H100 SDXL', alpha=0.8, edgecolors='white', linewidth=0.5)

# Draw Pareto frontier for each series
def draw_pareto(points, color, ls='--'):
    if not points:
        return
    sorted_pts = sorted(points, key=lambda p: p['energy'])
    pareto_e, pareto_c = [], []
    max_clip = -1
    for p in sorted_pts:
        if p['clip'] > max_clip:
            pareto_e.append(p['energy'])
            pareto_c.append(p['clip'])
            max_clip = p['clip']
    if len(pareto_e) > 1:
        ax.plot(pareto_e, pareto_c, color=color, linestyle=ls, linewidth=1.5, alpha=0.7)

draw_pareto(mac_points, MAC_COLOR)
draw_pareto(h100_sd_points, H100_COLOR)
draw_pareto(h100_sdxl_points, '#CC79A7')

# Annotate key points only (not every single one)
# Mac extremes
if mac_points:
    best_mac = max(mac_points, key=lambda p: p['clip'])
    cheapest_mac = min(mac_points, key=lambda p: p['energy'])
    ax.annotate(f"Mac {best_mac['res']}\n{best_mac['steps']}s", 
                (best_mac['energy'], best_mac['clip']),
                textcoords="offset points", xytext=(10, -10), fontsize=8, color=MAC_COLOR)
    ax.annotate(f"Mac {cheapest_mac['res']}\n{cheapest_mac['steps']}s",
                (cheapest_mac['energy'], cheapest_mac['clip']),
                textcoords="offset points", xytext=(-50, 10), fontsize=8, color=MAC_COLOR)

# H100 SDXL best
if h100_sdxl_points:
    best_sdxl = max(h100_sdxl_points, key=lambda p: p['clip'])
    ax.annotate(f"SDXL {best_sdxl['res']}\n{best_sdxl['steps']}s, CLIP={best_sdxl['clip']:.2f}",
                (best_sdxl['energy'], best_sdxl['clip']),
                textcoords="offset points", xytext=(10, 5), fontsize=8, color='#CC79A7')

ax.set_xlabel('Energy per Image (J)')
ax.set_ylabel('CLIP Score (Image-Text Alignment)')
ax.set_xscale('log')
ax.set_title('Energy-Quality Pareto Frontier')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0.15)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'fig7_energy_quality_pareto.png'))
plt.savefig(os.path.join(FIGURES, 'fig7_energy_quality_pareto.pdf'))
plt.close()
print("✅ fig7_energy_quality_pareto saved")

# ============================================================
# FIG 8: Power profile (colorblind-safe)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

categories = ['Text\n(256 tok)', 'Image\n(512², 20s)', 'Video\n(8f, 20s)', 'Music\n(small, 256)', 'SDXL\n(768², 20s)']
x = np.arange(len(categories))
width = 0.25

# Gather mean power for representative configs
def mean_power(data, filter_fn):
    filtered = [r for r in data if filter_fn(r)]
    if not filtered:
        return 0, 0
    powers = [r['avg_power_w'] for r in filtered]
    return np.mean(powers), np.std(powers)

mac_powers = [
    mean_power(mac_text, lambda r: get_tokens(r) == 256),
    mean_power(mac_img, lambda r: r.get('resolution') == '512x512' and r.get('steps', 0) == 20),
    mean_power(mac_vid, lambda r: r.get('frames', r.get('num_frames', 0)) == 8 and r.get('steps', 0) == 20),
    mean_power(mac_mus, lambda r: 'small' in str(r.get('model', '')).lower() and get_tokens(r) == 256),
    mean_power(mac_sdxl, lambda r: '768' in str(r.get('resolution', '')) and r.get('steps', 0) == 20),
]

a100_powers = [
    mean_power(a100_text, lambda r: get_tokens(r) == 256),
    mean_power(a100_img, lambda r: r.get('resolution') == '512x512' and r.get('steps', 0) == 20),
    mean_power(a100_vid, lambda r: r.get('frames', r.get('num_frames', 0)) == 8 and r.get('steps', 0) == 20),
    mean_power(a100_mus, lambda r: 'small' in str(r.get('model', '')).lower() and get_tokens(r) == 256),
    mean_power(a100_sdxl, lambda r: '768' in str(r.get('resolution', '')) and r.get('steps', 0) == 20),
]

h100_powers = [
    mean_power(h100_text, lambda r: get_tokens(r) == 256),
    mean_power(h100_img, lambda r: r.get('resolution') == '512x512' and r.get('steps', 0) == 20),
    mean_power(h100_vid, lambda r: r.get('frames', r.get('num_frames', 0)) == 8 and r.get('steps', 0) == 20),
    (0, 0),  # no music
    mean_power(h100_sdxl, lambda r: '768' in str(r.get('resolution', '')) and r.get('steps', 0) == 20),
]

bars_mac = ax.bar(x - width, [p[0] for p in mac_powers], width, yerr=[p[1] for p in mac_powers],
                  label='Mac M4 Pro', color=MAC_COLOR, capsize=3, alpha=0.85)
bars_a100 = ax.bar(x, [p[0] for p in a100_powers], width, yerr=[p[1] for p in a100_powers],
                   label='A100', color=A100_COLOR, capsize=3, alpha=0.85)
bars_h100 = ax.bar(x + width, [p[0] for p in h100_powers], width, yerr=[p[1] for p in h100_powers],
                   label='H100', color=H100_COLOR, capsize=3, alpha=0.85)

ax.set_xlabel('Workload')
ax.set_ylabel('Average Power Draw (W)')
ax.set_title('Power Draw Comparison Across Platforms')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add TDP lines
ax.axhline(y=22, color=MAC_COLOR, linestyle=':', alpha=0.5, linewidth=0.8)
ax.axhline(y=400, color=A100_COLOR, linestyle=':', alpha=0.5, linewidth=0.8)
ax.axhline(y=700, color=H100_COLOR, linestyle=':', alpha=0.5, linewidth=0.8)
ax.text(4.6, 25, 'Mac TDP (22W)', fontsize=8, color=MAC_COLOR, alpha=0.7)
ax.text(4.6, 410, 'A100 TDP (400W)', fontsize=8, color=A100_COLOR, alpha=0.7)
ax.text(4.6, 710, 'H100 TDP (700W)', fontsize=8, color=H100_COLOR, alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'fig8_power_profile_3hw.png'))
plt.savefig(os.path.join(FIGURES, 'fig8_power_profile_3hw.pdf'))
plt.close()
print("✅ fig8_power_profile_3hw saved")

# ============================================================
# FIG 6: Batched inference (colorblind-safe)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# A100 batched
a100_b_groups = group_by(a100_batch, lambda r: r.get('batch_size', 1))
a100_batches = sorted(a100_b_groups.keys())
a100_per_query = []
for b in a100_batches:
    runs = a100_b_groups[b]
    total_e = np.mean([r['total_energy_j'] for r in runs])
    a100_per_query.append(total_e / b)

# H100 batched
h100_b_groups = group_by(h100_batch, lambda r: r.get('batch_size', 1))
h100_batches = sorted(h100_b_groups.keys())
h100_per_query = []
for b in h100_batches:
    runs = h100_b_groups[b]
    total_e = np.mean([r['total_energy_j'] for r in runs])
    h100_per_query.append(total_e / b)

ax1.plot(a100_batches, a100_per_query, marker=A100_MARKER, color=A100_COLOR, label='A100', linewidth=1.5, markersize=7)
ax1.plot(h100_batches, h100_per_query, marker=H100_MARKER, color=H100_COLOR, label='H100', linewidth=1.5, markersize=7)
ax1.axhline(y=24.5, color=MAC_COLOR, linestyle='--', linewidth=1, label='Mac single-query (24.5J)')
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Energy per Query (J)')
ax1.set_title('(a) Per-Query Energy vs Batch Size')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Efficiency gain
if a100_per_query and a100_per_query[0] > 0:
    a100_gains = [a100_per_query[0] / pq for pq in a100_per_query]
    ax2.plot(a100_batches, a100_gains, marker=A100_MARKER, color=A100_COLOR, label='A100', linewidth=1.5, markersize=7)
if h100_per_query and h100_per_query[0] > 0:
    h100_gains = [h100_per_query[0] / pq for pq in h100_per_query]
    ax2.plot(h100_batches, h100_gains, marker=H100_MARKER, color=H100_COLOR, label='H100', linewidth=1.5, markersize=7)
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Efficiency Gain (×)')
ax2.set_title('(b) Efficiency Gain from Batching')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'fig6_batched_inference.png'))
plt.savefig(os.path.join(FIGURES, 'fig6_batched_inference.pdf'))
plt.close()
print("✅ fig6_batched_inference saved")

# ============================================================
# FIG 4: H100 vs A100 bar chart (colorblind-safe)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

modalities = ['Text\n(Phi-3)', 'Image\n(SD v1.5)', 'Video\n(AnimateDiff)', 'Image\n(SDXL)', 'Batched\n(Phi-3)']

# Compute H100/A100 ratios for common configs
def compute_ratio(a100_data, h100_data, key_fn):
    a100_g = group_by(a100_data, key_fn)
    h100_g = group_by(h100_data, key_fn)
    common = set(a100_g.keys()) & set(h100_g.keys())
    if not common:
        return 0
    ratios = []
    for k in common:
        a_e = np.mean([r['total_energy_j'] for r in a100_g[k]])
        h_e = np.mean([r['total_energy_j'] for r in h100_g[k]])
        ratios.append(h_e / a_e)
    return np.mean(ratios)

ratios = [
    compute_ratio(a100_text, h100_text, get_tokens),
    compute_ratio(a100_img, h100_img, lambda r: get_res_steps(r) if r.get('steps', 0) == 20 else None),
    compute_ratio(a100_vid, h100_vid, lambda r: get_frames_steps(r) if r.get('steps', 0) == 20 else None),
    compute_ratio(a100_sdxl, h100_sdxl, lambda r: get_res_steps(r) if r.get('steps', 0) == 20 else None),
    compute_ratio(a100_batch, h100_batch, lambda r: r.get('batch_size', 1)),
]

colors_bar = [A100_COLOR if r >= 0.8 else H100_COLOR for r in ratios]
bars = ax.bar(modalities, ratios, color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
ax.set_ylabel('Energy Ratio (H100 / A100)')
ax.set_title('H100 vs A100 Energy Efficiency')

# Add percentage labels
for bar, ratio in zip(bars, ratios):
    pct = (1 - ratio) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{pct:.0f}% less' if pct > 0 else f'{-pct:.0f}% more',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylim(0, 1.2)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, 'fig4_h100_vs_a100.png'))
plt.savefig(os.path.join(FIGURES, 'fig4_h100_vs_a100.pdf'))
plt.close()
print("✅ fig4_h100_vs_a100 saved")

print("\n🎉 All figures regenerated with colorblind-safe palette!")
