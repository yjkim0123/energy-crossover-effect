#!/usr/bin/env python3
"""
Cross-Hardware Generative AI Energy Benchmarking Analysis
3 platforms: Mac mini M4 Pro, NVIDIA A100, NVIDIA H100
Generates summary stats, figures, scaling laws, and key numbers.
"""

import json
import os
import sys
import warnings
import numpy as np
from scipy.optimize import curve_fit
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────
BASE = os.path.expanduser("~/Documents/project_energy")
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(BASE, "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────
COLORS = {"Mac": "#2196F3", "A100": "#F44336", "H100": "#4CAF50"}
MARKERS = {"Mac": "o", "A100": "s", "H100": "^"}
HW_LABELS = {
    "Mac": "Mac mini M4 Pro",
    "A100": "A100-SXM4-40GB",
    "H100": "H100-80GB-HBM3",
}

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})


# ── Helper: load JSON ─────────────────────────────────────────────
def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def save_fig(fig, name):
    fig.savefig(os.path.join(FIGURES, name + ".png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES, name + ".pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}.png/.pdf")


# ══════════════════════════════════════════════════════════════════
#  LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════
print("Loading data files...")

# --- Text (Phi-3) ---
text_mac = load("exp1_text.json")
text_a100 = load("exp1_text_A100.json")
text_h100 = load("text_phi3_H100.json")

# --- Image SD1.5 ---
img_mac = load("exp2_image.json") + load("exp2_image_extra_Mac.json")
img_a100 = load("exp2_image_A100.json") + load("exp2_image_extra_A100.json")
img_h100 = load("image_sd15_H100.json")

# --- Video AnimateDiff ---
vid_mac = load("exp3_video.json") + load("exp3_video_extra_Mac.json")
vid_a100 = load("exp3_video_A100.json") + load("exp3_video_extra_A100.json")
vid_h100 = load("video_animatediff_H100.json")

# --- Music MusicGen ---
mus_mac = load("exp4_music.json")
mus_a100 = load("exp4_music_A100.json")
# No H100 music data

# --- SDXL ---
sdxl_mac = load("exp5_sdxl_Mac.json")
sdxl_a100 = load("exp5_sdxl_A100.json")
sdxl_h100 = load("image_sdxl_H100.json")

# --- Batched ---
batch_a100 = load("exp7_batched_A100.json")
batch_h100 = load("batched_phi3_H100.json")

# --- Quality (CLIP) ---
qual_mac = load("quality_image_sd15_Mac.json")
# H100 SD1.5 and SDXL also have clip scores

total_runs = (len(text_mac) + len(text_a100) + len(text_h100)
              + len(img_mac) + len(img_a100) + len(img_h100)
              + len(vid_mac) + len(vid_a100) + len(vid_h100)
              + len(mus_mac) + len(mus_a100)
              + len(sdxl_mac) + len(sdxl_a100) + len(sdxl_h100)
              + len(batch_a100) + len(batch_h100)
              + len(qual_mac))
print(f"Total records loaded: {total_runs}")


# ══════════════════════════════════════════════════════════════════
#  HELPER: aggregate by config
# ══════════════════════════════════════════════════════════════════
def agg_by_key(data, key_fn, energy_key='total_energy_j'):
    """Group records by key_fn, compute mean/std energy, mean power, mean duration.
    Skips records missing the energy_key."""
    groups = defaultdict(list)
    for r in data:
        if energy_key not in r or r[energy_key] is None:
            continue  # skip incomplete records
        k = key_fn(r)
        groups[k].append(r)
    result = {}
    for k, recs in sorted(groups.items()):
        energies = [r[energy_key] for r in recs]
        powers = [r['avg_power_w'] for r in recs if 'avg_power_w' in r]
        durations = [r.get('duration_sec', r.get('generation_time_sec', 0)) for r in recs]
        clips = [r['clip_score'] for r in recs if 'clip_score' in r and r.get('clip_score') is not None]
        result[k] = {
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'mean_power': np.mean(powers) if powers else 0,
            'mean_duration': np.mean(durations),
            'mean_clip': np.mean(clips) if clips else None,
            'count': len(recs),
        }
    return result


def res_to_pixels(res_str):
    """'512x512' -> 262144"""
    parts = res_str.split('x')
    return int(parts[0]) * int(parts[1])


# ══════════════════════════════════════════════════════════════════
#  1. SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════
print("\n=== Computing Summary Statistics ===")

summary = {}

# Text
for hw, data in [("Mac", text_mac), ("A100", text_a100), ("H100", text_h100)]:
    tok_key = 'max_tokens' if 'max_tokens' in data[0] else 'max_tokens'
    agg = agg_by_key(data, lambda r: r.get('max_tokens', r.get('actual_tokens', 0)))
    for k, v in agg.items():
        summary[f"text|{hw}|tokens={k}"] = v

# Image SD1.5
for hw, data in [("Mac", img_mac), ("A100", img_a100), ("H100", img_h100)]:
    agg = agg_by_key(data, lambda r: (r['resolution'], r['steps']))
    for k, v in agg.items():
        summary[f"image_sd15|{hw}|res={k[0]},steps={k[1]}"] = v

# Video
for hw, data in [("Mac", vid_mac), ("A100", vid_a100), ("H100", vid_h100)]:
    agg = agg_by_key(data, lambda r: (r['frames'], r['steps']))
    for k, v in agg.items():
        summary[f"video|{hw}|frames={k[0]},steps={k[1]}"] = v

# Music
for hw, data in [("Mac", mus_mac), ("A100", mus_a100)]:
    agg = agg_by_key(data, lambda r: r['max_tokens'])
    for k, v in agg.items():
        summary[f"music|{hw}|tokens={k}"] = v

# SDXL
for hw, data in [("Mac", sdxl_mac), ("A100", sdxl_a100), ("H100", sdxl_h100)]:
    agg = agg_by_key(data, lambda r: (r['resolution'], r['steps']))
    for k, v in agg.items():
        summary[f"sdxl|{hw}|res={k[0]},steps={k[1]}"] = v

# Batched
for hw, data in [("A100", batch_a100), ("H100", batch_h100)]:
    agg = agg_by_key(data, lambda r: r['batch_size'], energy_key='per_query_energy_j')
    for k, v in agg.items():
        # re-compute with per_query_energy
        pq = [r['per_query_energy_j'] for r in data if r['batch_size'] == k]
        summary[f"batched|{hw}|batch={k}"] = {
            **v,
            'mean_per_query_energy': np.mean(pq),
            'std_per_query_energy': np.std(pq),
        }

with open(os.path.join(RESULTS, "cross_hardware_summary.json"), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("  Saved cross_hardware_summary.json")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 1: 4-panel comparison (text, image, video, music)
# ══════════════════════════════════════════════════════════════════
print("\n=== Figure 1: 3-Hardware Comparison (4 panels) ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# --- Panel A: Text ---
ax = axes[0, 0]
for hw, data, label in [("Mac", text_mac, HW_LABELS["Mac"]),
                         ("A100", text_a100, HW_LABELS["A100"]),
                         ("H100", text_h100, HW_LABELS["H100"])]:
    agg = agg_by_key(data, lambda r: r['max_tokens'])
    xs = sorted(agg.keys())
    means = [agg[x]['mean_energy'] for x in xs]
    stds = [agg[x]['std_energy'] for x in xs]
    ax.errorbar(xs, means, yerr=stds, marker=MARKERS[hw], color=COLORS[hw],
                label=label, capsize=4, linewidth=2, markersize=7)
ax.set_xlabel("Max Tokens")
ax.set_ylabel("Energy (J)")
ax.set_title("(a) Text Generation (Phi-3)")
ax.legend(fontsize=10)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')

# --- Panel B: Image SD1.5 (vary resolution, fixed steps=20) ---
ax = axes[0, 1]
for hw, data, label in [("Mac", img_mac, HW_LABELS["Mac"]),
                         ("A100", img_a100, HW_LABELS["A100"]),
                         ("H100", img_h100, HW_LABELS["H100"])]:
    filtered = [r for r in data if r['steps'] == 20]
    if not filtered:
        continue
    agg = agg_by_key(filtered, lambda r: res_to_pixels(r['resolution']))
    xs = sorted(agg.keys())
    means = [agg[x]['mean_energy'] for x in xs]
    stds = [agg[x]['std_energy'] for x in xs]
    ax.errorbar(xs, means, yerr=stds, marker=MARKERS[hw], color=COLORS[hw],
                label=label, capsize=4, linewidth=2, markersize=7)
ax.set_xlabel("Resolution (total pixels)")
ax.set_ylabel("Energy (J)")
ax.set_title("(b) Image Generation (SD v1.5, 20 steps)")
ax.legend(fontsize=10)
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, which='both')

# --- Panel C: Video (vary frames, fixed steps=20) ---
ax = axes[1, 0]
for hw, data, label in [("Mac", vid_mac, HW_LABELS["Mac"]),
                         ("A100", vid_a100, HW_LABELS["A100"]),
                         ("H100", vid_h100, HW_LABELS["H100"])]:
    filtered = [r for r in data if r['steps'] == 20]
    if not filtered:
        # H100 might have steps=10,20,30
        steps_avail = sorted(set(r['steps'] for r in data))
        # pick the middle one
        mid_step = steps_avail[len(steps_avail)//2]
        filtered = [r for r in data if r['steps'] == mid_step]
        label += f" (steps={mid_step})"
    agg = agg_by_key(filtered, lambda r: r['frames'])
    xs = sorted(agg.keys())
    means = [agg[x]['mean_energy'] for x in xs]
    stds = [agg[x]['std_energy'] for x in xs]
    ax.errorbar(xs, means, yerr=stds, marker=MARKERS[hw], color=COLORS[hw],
                label=label, capsize=4, linewidth=2, markersize=7)
ax.set_xlabel("Number of Frames")
ax.set_ylabel("Energy (J)")
ax.set_title("(c) Video Generation (AnimateDiff)")
ax.legend(fontsize=10)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# --- Panel D: Music (vary tokens) ---
ax = axes[1, 1]
for hw, data, label in [("Mac", mus_mac, HW_LABELS["Mac"]),
                         ("A100", mus_a100, HW_LABELS["A100"])]:
    agg = agg_by_key(data, lambda r: r['max_tokens'])
    xs = sorted(agg.keys())
    means = [agg[x]['mean_energy'] for x in xs]
    stds = [agg[x]['std_energy'] for x in xs]
    ax.errorbar(xs, means, yerr=stds, marker=MARKERS[hw], color=COLORS[hw],
                label=label, capsize=4, linewidth=2, markersize=7)
ax.set_xlabel("Max Tokens (Audio Length)")
ax.set_ylabel("Energy (J)")
ax.set_title("(d) Music Generation (MusicGen)")
ax.legend(fontsize=10)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

fig.suptitle("Cross-Hardware Energy Comparison Across Generative AI Modalities", fontsize=15, y=1.01)
fig.tight_layout()
save_fig(fig, "fig1_3hw_comparison")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 2: Efficiency Ratio Plot
# ══════════════════════════════════════════════════════════════════
print("\n=== Figure 2: Efficiency Ratio ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

modalities = [
    ("Text (Phi-3)", text_mac, text_a100, text_h100, lambda r: r['max_tokens'], "Max Tokens"),
    ("Image (SD v1.5, 512×512)",
     [r for r in img_mac if r['resolution'] == '512x512'],
     [r for r in img_a100 if r['resolution'] == '512x512'],
     [r for r in img_h100 if r['resolution'] == '512x512'],
     lambda r: r['steps'], "Diffusion Steps"),
    ("Video (AnimateDiff, steps=20)",
     [r for r in vid_mac if r['steps'] == 20],
     [r for r in vid_a100 if r['steps'] == 20],
     [r for r in vid_h100 if r['steps'] == 20],
     lambda r: r['frames'], "Frames"),
    ("Music (MusicGen)",
     mus_mac, mus_a100, None,
     lambda r: r['max_tokens'], "Max Tokens"),
]

for idx, (title, mac_d, a100_d, h100_d, key_fn, xlabel) in enumerate(modalities):
    ax = axes[idx // 2, idx % 2]

    mac_agg = agg_by_key(mac_d, key_fn) if mac_d else {}
    a100_agg = agg_by_key(a100_d, key_fn) if a100_d else {}
    h100_agg = agg_by_key(h100_d, key_fn) if h100_d else {}

    common_keys = sorted(set(mac_agg.keys()) & set(a100_agg.keys()))
    if common_keys:
        xs = common_keys
        ratios_a100 = [a100_agg[x]['mean_energy'] / mac_agg[x]['mean_energy'] for x in xs]
        ax.plot(xs, ratios_a100, marker='s', color=COLORS['A100'], linewidth=2,
                markersize=8, label=f"A100 / Mac")

    common_h100 = sorted(set(mac_agg.keys()) & set(h100_agg.keys())) if h100_agg else []
    if common_h100:
        xs_h = common_h100
        ratios_h100 = [h100_agg[x]['mean_energy'] / mac_agg[x]['mean_energy'] for x in xs_h]
        ax.plot(xs_h, ratios_h100, marker='^', color=COLORS['H100'], linewidth=2,
                markersize=8, label=f"H100 / Mac")

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    # green zone below 1.0
    all_xs = sorted(set(list(common_keys) + list(common_h100)))
    if all_xs:
        ax.fill_between([min(all_xs)*0.8, max(all_xs)*1.2], 0, 1.0,
                        color='#4CAF50', alpha=0.08, label='GPU more efficient')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy Ratio (GPU / Mac)")
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle("Energy Efficiency Ratios: GPU vs Edge (Mac mini M4 Pro)", fontsize=15, y=1.01)
fig.tight_layout()
save_fig(fig, "fig2_efficiency_ratio_3hw")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 3: Scaling Laws E = α·N^β
# ══════════════════════════════════════════════════════════════════
print("\n=== Figure 3: Scaling Laws ===")

def power_law(x, alpha, beta):
    return alpha * np.power(x, beta)


def fit_power_law(xs, ys):
    """Fit E = alpha * N^beta, return alpha, beta, R²."""
    try:
        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        popt, _ = curve_fit(power_law, xs, ys, p0=[1.0, 1.0], maxfev=10000)
        alpha, beta = popt
        y_pred = power_law(xs, alpha, beta)
        ss_res = np.sum((ys - y_pred) ** 2)
        ss_tot = np.sum((ys - np.mean(ys)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return alpha, beta, r2
    except Exception as e:
        return None, None, None


scaling_laws = {}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

scaling_configs = [
    ("Text (Phi-3)", text_mac, text_a100, text_h100,
     lambda r: r['max_tokens'], "Max Tokens", "text"),
    ("Image (SD v1.5, 512×512)",
     [r for r in img_mac if r['resolution'] == '512x512'],
     [r for r in img_a100 if r['resolution'] == '512x512'],
     [r for r in img_h100 if r['resolution'] == '512x512'],
     lambda r: r['steps'], "Diffusion Steps", "image_sd15"),
    ("Video (AnimateDiff, steps=20)",
     [r for r in vid_mac if r['steps'] == 20],
     [r for r in vid_a100 if r['steps'] == 20],
     [r for r in vid_h100 if r['steps'] == 20],
     lambda r: r['frames'], "Frames", "video"),
    ("Music (MusicGen)",
     mus_mac, mus_a100, None,
     lambda r: r['max_tokens'], "Max Tokens", "music"),
]

for idx, (title, mac_d, a100_d, h100_d, key_fn, xlabel, mod_name) in enumerate(scaling_configs):
    ax = axes[idx // 2, idx % 2]

    for hw, data, label in [("Mac", mac_d, HW_LABELS["Mac"]),
                             ("A100", a100_d, HW_LABELS["A100"]),
                             ("H100", h100_d, HW_LABELS["H100"])]:
        if data is None or len(data) == 0:
            continue
        agg = agg_by_key(data, key_fn)
        xs = sorted(agg.keys())
        means = [agg[x]['mean_energy'] for x in xs]

        # Scatter
        ax.scatter(xs, means, marker=MARKERS[hw], color=COLORS[hw], s=60, zorder=5)

        # Fit
        alpha, beta, r2 = fit_power_law(xs, means)
        if alpha is not None:
            scaling_laws[f"{mod_name}|{hw}"] = {
                'alpha': float(alpha), 'beta': float(beta), 'R2': float(r2)
            }
            x_fit = np.linspace(min(xs) * 0.8, max(xs) * 1.2, 200)
            y_fit = power_law(x_fit, alpha, beta)
            ax.plot(x_fit, y_fit, color=COLORS[hw], linewidth=2, linestyle='--',
                    label=f"{label}: α={alpha:.2f}, β={beta:.2f}, R²={r2:.3f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy (J)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')

fig.suptitle("Scaling Laws: E = α·N^β Across Hardware Platforms", fontsize=15, y=1.01)
fig.tight_layout()
save_fig(fig, "fig3_scaling_laws_3hw")


# ── Compute crossover points ──────────────────────────────────────
def crossover_point(alpha1, beta1, alpha2, beta2):
    """Find N where alpha1*N^beta1 = alpha2*N^beta2 → N = (alpha1/alpha2)^(1/(beta2-beta1))"""
    if beta2 == beta1:
        return None
    try:
        ratio = alpha1 / alpha2
        exp = 1.0 / (beta2 - beta1)
        if ratio <= 0:
            return None
        return ratio ** exp
    except:
        return None


for mod in ["text", "image_sd15", "video", "music"]:
    pairs = [("Mac", "A100"), ("Mac", "H100"), ("A100", "H100")]
    for hw1, hw2 in pairs:
        k1 = f"{mod}|{hw1}"
        k2 = f"{mod}|{hw2}"
        if k1 in scaling_laws and k2 in scaling_laws:
            cp = crossover_point(
                scaling_laws[k1]['alpha'], scaling_laws[k1]['beta'],
                scaling_laws[k2]['alpha'], scaling_laws[k2]['beta']
            )
            scaling_laws[f"crossover|{mod}|{hw1}_vs_{hw2}"] = {
                'crossover_N': float(cp) if cp else None
            }

with open(os.path.join(RESULTS, "scaling_laws_3hw.json"), 'w') as f:
    json.dump(scaling_laws, f, indent=2)
print("  Saved scaling_laws_3hw.json")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 4: H100 vs A100 Direct Comparison
# ══════════════════════════════════════════════════════════════════
print("\n=== Figure 4: H100 vs A100 ===")

fig, ax = plt.subplots(figsize=(10, 6))

modality_configs = []

# Text: average across all token lengths
text_a100_mean = np.mean([r['total_energy_j'] for r in text_a100])
text_h100_mean = np.mean([r['total_energy_j'] for r in text_h100])
modality_configs.append(("Text\n(Phi-3)", text_h100_mean / text_a100_mean,
                          np.mean([r['generation_time_sec'] for r in text_h100]) /
                          np.mean([r['generation_time_sec'] for r in text_a100])))

# Image SD1.5: common configs
common_img_configs = set((r['resolution'], r['steps']) for r in img_a100) & \
                     set((r['resolution'], r['steps']) for r in img_h100)
if common_img_configs:
    a100_e = np.mean([r['total_energy_j'] for r in img_a100 if (r['resolution'], r['steps']) in common_img_configs])
    h100_e = np.mean([r['total_energy_j'] for r in img_h100 if (r['resolution'], r['steps']) in common_img_configs])
    modality_configs.append(("Image\n(SD v1.5)", h100_e / a100_e, None))

# Video
common_vid = set((r['frames'], r['steps']) for r in vid_a100) & \
             set((r['frames'], r['steps']) for r in vid_h100)
if common_vid:
    a100_e = np.mean([r['total_energy_j'] for r in vid_a100 if (r['frames'], r['steps']) in common_vid])
    h100_e = np.mean([r['total_energy_j'] for r in vid_h100 if (r['frames'], r['steps']) in common_vid])
    modality_configs.append(("Video\n(AnimateDiff)", h100_e / a100_e, None))

# SDXL
common_sdxl = set((r['resolution'], r['steps']) for r in sdxl_a100) & \
              set((r['resolution'], r['steps']) for r in sdxl_h100)
if common_sdxl:
    a100_e = np.mean([r['total_energy_j'] for r in sdxl_a100 if (r['resolution'], r['steps']) in common_sdxl])
    h100_e = np.mean([r['total_energy_j'] for r in sdxl_h100 if (r['resolution'], r['steps']) in common_sdxl])
    modality_configs.append(("Image\n(SDXL)", h100_e / a100_e, None))

names = [m[0] for m in modality_configs]
ratios = [m[1] for m in modality_configs]
bar_colors = [COLORS['H100'] if r < 1 else COLORS['A100'] for r in ratios]

bars = ax.bar(names, ratios, color=bar_colors, edgecolor='black', linewidth=0.8, width=0.6, alpha=0.85)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Parity (H100 = A100)')
ax.fill_between([-0.5, len(names)-0.5], 0, 1.0, color=COLORS['H100'], alpha=0.06)

for bar, ratio in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{ratio:.2f}×', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel("Energy Ratio (H100 / A100)")
ax.set_title("H100 vs A100: Energy Consumption by Modality")
ax.legend()
ax.set_ylim(0, 1.15)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
save_fig(fig, "fig4_h100_vs_a100")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 5: SDXL 3-Hardware Comparison
# ══════════════════════════════════════════════════════════════════
print("\n=== Figure 5: SDXL Comparison ===")

fig, ax = plt.subplots(figsize=(12, 7))

for hw, data, label in [("Mac", sdxl_mac, HW_LABELS["Mac"]),
                         ("A100", sdxl_a100, HW_LABELS["A100"]),
                         ("H100", sdxl_h100, HW_LABELS["H100"])]:
    agg = agg_by_key(data, lambda r: (r['resolution'], r['steps']))
    # Create a composite x-axis: resolution × steps as a string
    configs = sorted(agg.keys(), key=lambda k: (res_to_pixels(k[0]), k[1]))
    x_labels = [f"{c[0]}\n{c[1]}s" for c in configs]
    means = [agg[c]['mean_energy'] for c in configs]
    stds = [agg[c]['std_energy'] for c in configs]

    x_pos = np.arange(len(configs))
    ax.errorbar(x_pos, means, yerr=stds, marker=MARKERS[hw], color=COLORS[hw],
                label=label, capsize=4, linewidth=2, markersize=7)

ax.set_xticks(range(len(x_labels)))
# Use the longest set of configs for x labels
all_configs_sets = []
for hw, data in [("Mac", sdxl_mac), ("A100", sdxl_a100), ("H100", sdxl_h100)]:
    agg = agg_by_key(data, lambda r: (r['resolution'], r['steps']))
    all_configs_sets.append(set(agg.keys()))

# Use H100 as it has the most configs
h100_agg = agg_by_key(sdxl_h100, lambda r: (r['resolution'], r['steps']))
h100_configs = sorted(h100_agg.keys(), key=lambda k: (res_to_pixels(k[0]), k[1]))
h100_labels = [f"{c[0]}\n{c[1]}s" for c in h100_configs]

# Re-plot properly with aligned x-axis
fig, ax = plt.subplots(figsize=(14, 7))

# Get union of all configs
all_configs = set()
for hw, data in [("Mac", sdxl_mac), ("A100", sdxl_a100), ("H100", sdxl_h100)]:
    for r in data:
        all_configs.add((r['resolution'], r['steps']))
all_configs = sorted(all_configs, key=lambda k: (res_to_pixels(k[0]), k[1]))
config_labels = [f"{c[0]}\n{c[1]}s" for c in all_configs]
x_pos = np.arange(len(all_configs))

for hw, data, label in [("Mac", sdxl_mac, HW_LABELS["Mac"]),
                         ("A100", sdxl_a100, HW_LABELS["A100"]),
                         ("H100", sdxl_h100, HW_LABELS["H100"])]:
    agg = agg_by_key(data, lambda r: (r['resolution'], r['steps']))
    xs, means, stds = [], [], []
    for i, c in enumerate(all_configs):
        if c in agg:
            xs.append(i)
            means.append(agg[c]['mean_energy'])
            stds.append(agg[c]['std_energy'])
    ax.errorbar(xs, means, yerr=stds, marker=MARKERS[hw], color=COLORS[hw],
                label=label, capsize=4, linewidth=2, markersize=7)

ax.set_xticks(x_pos)
ax.set_xticklabels(config_labels, fontsize=9, rotation=45, ha='right')
ax.set_xlabel("Resolution × Steps")
ax.set_ylabel("Energy (J)")
ax.set_title("SDXL Image Generation: Cross-Hardware Energy Comparison")
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig5_sdxl_3hw")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 6: Batched Inference
# ══════════════════════════════════════════════════════════════════
print("\n=== Figure 6: Batched Inference ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Per-query energy
ax = axes[0]
for hw, data, label in [("A100", batch_a100, HW_LABELS["A100"]),
                         ("H100", batch_h100, HW_LABELS["H100"])]:
    agg = agg_by_key(data, lambda r: r['batch_size'], energy_key='per_query_energy_j')
    xs = sorted(agg.keys())
    means = [agg[x]['mean_energy'] for x in xs]
    stds = [agg[x]['std_energy'] for x in xs]
    ax.errorbar(xs, means, yerr=stds, marker=MARKERS[hw], color=COLORS[hw],
                label=label, capsize=4, linewidth=2, markersize=8)

ax.set_xlabel("Batch Size")
ax.set_ylabel("Per-Query Energy (J)")
ax.set_title("(a) Per-Query Energy vs Batch Size")
ax.set_xscale('log', base=2)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Efficiency gain (normalized to batch=1)
ax = axes[1]
for hw, data, label in [("A100", batch_a100, HW_LABELS["A100"]),
                         ("H100", batch_h100, HW_LABELS["H100"])]:
    agg = agg_by_key(data, lambda r: r['batch_size'], energy_key='per_query_energy_j')
    xs = sorted(agg.keys())
    means = [agg[x]['mean_energy'] for x in xs]
    base = means[0]
    normalized = [base / m for m in means]  # efficiency gain
    ax.plot(xs, normalized, marker=MARKERS[hw], color=COLORS[hw],
            label=label, linewidth=2, markersize=8)

ax.set_xlabel("Batch Size")
ax.set_ylabel("Efficiency Gain (×)")
ax.set_title("(b) Batching Efficiency Gain (relative to batch=1)")
ax.set_xscale('log', base=2)
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("Batched Inference: A100 vs H100 (Phi-3, 256 tokens)", fontsize=15, y=1.02)
fig.tight_layout()
save_fig(fig, "fig6_batched_inference")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 7: Energy-Quality Pareto Frontier
# ══════════════════════════════════════════════════════════════════
print("\n=== Figure 7: Energy-Quality Pareto ===")

fig, ax = plt.subplots(figsize=(12, 8))

# Mac quality data
qual_mac_agg = agg_by_key(qual_mac, lambda r: (r['resolution'], r['steps']))
# H100 SD1.5 has clip scores
h100_sd15_agg = agg_by_key(img_h100, lambda r: (r['resolution'], r['steps']))
# H100 SDXL has clip scores
h100_sdxl_agg = agg_by_key(sdxl_h100, lambda r: (r['resolution'], r['steps']))


def plot_pareto(ax, agg, color, marker, label, marker_size=60):
    """Plot points and Pareto frontier."""
    points = []
    for k, v in agg.items():
        if v['mean_clip'] is not None and v['mean_clip'] > 0:
            points.append((v['mean_energy'], v['mean_clip'], k))

    if not points:
        return

    energies = [p[0] for p in points]
    clips = [p[1] for p in points]

    ax.scatter(energies, clips, marker=marker, color=color, s=marker_size,
               label=label, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=5)

    # Annotate with config
    for e, c, cfg in points:
        res = cfg[0].split('x')[0]
        steps = cfg[1]
        ax.annotate(f'{res}p/{steps}s', (e, c), fontsize=7, alpha=0.6,
                    xytext=(5, 5), textcoords='offset points')

    # Compute Pareto front (maximize CLIP, minimize energy)
    sorted_pts = sorted(points, key=lambda p: p[0])  # sort by energy
    pareto = [sorted_pts[0]]
    for p in sorted_pts[1:]:
        if p[1] > pareto[-1][1]:  # higher CLIP
            pareto.append(p)

    if len(pareto) > 1:
        pe = [p[0] for p in pareto]
        pc = [p[1] for p in pareto]
        ax.plot(pe, pc, color=color, linewidth=2, linestyle='--', alpha=0.6)


plot_pareto(ax, qual_mac_agg, COLORS['Mac'], MARKERS['Mac'], f"Mac (SD v1.5)")
plot_pareto(ax, h100_sd15_agg, COLORS['H100'], MARKERS['H100'], f"H100 (SD v1.5)")
# Also add H100 SDXL
plot_pareto(ax, h100_sdxl_agg, '#8BC34A', 'D', f"H100 (SDXL)")

ax.set_xlabel("Energy (J)")
ax.set_ylabel("CLIP Score")
ax.set_title("Energy–Quality Pareto Frontier: Image Generation")
ax.legend()
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig(fig, "fig7_energy_quality_pareto")


# ══════════════════════════════════════════════════════════════════
#  FIGURE 8: Power Profile Comparison
# ══════════════════════════════════════════════════════════════════
print("\n=== Figure 8: Power Profile ===")

fig, ax = plt.subplots(figsize=(12, 7))

def safe_mean_power(data):
    vals = [r['avg_power_w'] for r in data if 'avg_power_w' in r]
    return np.mean(vals) if vals else 0

modalities_power = {
    'Text\n(Phi-3)': {
        'Mac': safe_mean_power(text_mac),
        'A100': safe_mean_power(text_a100),
        'H100': safe_mean_power(text_h100),
    },
    'Image\n(SD v1.5)': {
        'Mac': safe_mean_power(img_mac),
        'A100': safe_mean_power(img_a100),
        'H100': safe_mean_power(img_h100),
    },
    'Video\n(AnimateDiff)': {
        'Mac': safe_mean_power(vid_mac),
        'A100': safe_mean_power(vid_a100),
        'H100': safe_mean_power(vid_h100),
    },
    'Music\n(MusicGen)': {
        'Mac': safe_mean_power(mus_mac),
        'A100': safe_mean_power(mus_a100),
    },
    'Image\n(SDXL)': {
        'Mac': safe_mean_power(sdxl_mac),
        'A100': safe_mean_power(sdxl_a100),
        'H100': safe_mean_power(sdxl_h100),
    },
}

mod_names = list(modalities_power.keys())
x = np.arange(len(mod_names))
width = 0.25

for i, (hw, offset) in enumerate([("Mac", -width), ("A100", 0), ("H100", width)]):
    vals = [modalities_power[m].get(hw, 0) for m in mod_names]
    bars = ax.bar(x + offset, vals, width, label=HW_LABELS[hw], color=COLORS[hw],
                  edgecolor='black', linewidth=0.5, alpha=0.85)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{v:.0f}W', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(mod_names)
ax.set_ylabel("Average Power Draw (W)")
ax.set_title("Average Power Consumption by Modality and Hardware")
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
save_fig(fig, "fig8_power_profile_3hw")


# ══════════════════════════════════════════════════════════════════
#  4. KEY NUMBERS FOR PAPER
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  KEY NUMBERS FOR PAPER")
print("="*70)

print(f"\n📊 Total experimental runs across all 3 platforms: {total_runs}")

# Per-modality summaries
print("\n─── Per-Modality Energy Summary ───")

def modality_summary(name, mac_data, a100_data, h100_data):
    mac_e = np.mean([r['total_energy_j'] for r in mac_data if 'total_energy_j' in r]) if mac_data else None
    a100_e = np.mean([r['total_energy_j'] for r in a100_data if 'total_energy_j' in r]) if a100_data else None
    h100_e = np.mean([r['total_energy_j'] for r in h100_data if 'total_energy_j' in r]) if h100_data else None

    print(f"\n  {name}:")
    if mac_e: print(f"    Mac M4 Pro:  {mac_e:.2f} J (mean)")
    if a100_e: print(f"    A100:        {a100_e:.2f} J (mean)")
    if h100_e: print(f"    H100:        {h100_e:.2f} J (mean)")
    if mac_e and a100_e: print(f"    A100/Mac ratio: {a100_e/mac_e:.2f}×")
    if mac_e and h100_e: print(f"    H100/Mac ratio: {h100_e/mac_e:.2f}×")
    if a100_e and h100_e: print(f"    H100/A100 ratio: {h100_e/a100_e:.2f}×")

    return mac_e, a100_e, h100_e

modality_summary("Text Generation (Phi-3)", text_mac, text_a100, text_h100)
modality_summary("Image Generation (SD v1.5)", img_mac, img_a100, img_h100)
modality_summary("Video Generation (AnimateDiff)", vid_mac, vid_a100, vid_h100)
modality_summary("Music Generation (MusicGen)", mus_mac, mus_a100, None)
modality_summary("Image Generation (SDXL)", sdxl_mac, sdxl_a100, sdxl_h100)

# Crossover points
print("\n─── Scaling Law Crossover Points ───")
for key, val in scaling_laws.items():
    if key.startswith("crossover"):
        parts = key.split("|")
        mod = parts[1]
        pair = parts[2]
        cp = val.get('crossover_N')
        if cp is not None and cp > 0:
            print(f"  {mod} ({pair}): N = {cp:.1f}")
        else:
            print(f"  {mod} ({pair}): No crossover (one platform always more efficient)")

# H100 vs A100 speedup
print("\n─── H100 vs A100 Speedup ───")
for name, a100_d, h100_d in [
    ("Text", text_a100, text_h100),
    ("Image SD1.5", img_a100, img_h100),
    ("Video", vid_a100, vid_h100),
    ("SDXL", sdxl_a100, sdxl_h100)
]:
    a100_t = np.mean([r.get('generation_time_sec', r.get('duration_sec', 0)) for r in a100_d if 'generation_time_sec' in r or 'duration_sec' in r])
    h100_t = np.mean([r.get('generation_time_sec', r.get('duration_sec', 0)) for r in h100_d if 'generation_time_sec' in r or 'duration_sec' in r])
    a100_e = np.mean([r['total_energy_j'] for r in a100_d if 'total_energy_j' in r])
    h100_e = np.mean([r['total_energy_j'] for r in h100_d if 'total_energy_j' in r])
    print(f"  {name}: H100 is {a100_t/h100_t:.2f}× faster, uses {h100_e/a100_e:.2f}× energy")

# Batched inference
print("\n─── Batched Inference Efficiency ───")
for hw, data, label in [("A100", batch_a100, "A100"), ("H100", batch_h100, "H100")]:
    agg = agg_by_key(data, lambda r: r['batch_size'], energy_key='per_query_energy_j')
    batch_sizes = sorted(agg.keys())
    b1 = agg[batch_sizes[0]]['mean_energy']
    b_max = agg[batch_sizes[-1]]['mean_energy']
    print(f"  {label}: batch=1 → {b1:.2f} J/query, batch={batch_sizes[-1]} → {b_max:.2f} J/query")
    print(f"    Efficiency gain: {b1/b_max:.2f}×")

# Power draw summary
print("\n─── Average Power Draw ───")
for name, datasets in [
    ("Text", [("Mac", text_mac), ("A100", text_a100), ("H100", text_h100)]),
    ("Image SD1.5", [("Mac", img_mac), ("A100", img_a100), ("H100", img_h100)]),
    ("Video", [("Mac", vid_mac), ("A100", vid_a100), ("H100", vid_h100)]),
    ("Music", [("Mac", mus_mac), ("A100", mus_a100)]),
    ("SDXL", [("Mac", sdxl_mac), ("A100", sdxl_a100), ("H100", sdxl_h100)]),
]:
    powers = []
    for hw, d in datasets:
        p = safe_mean_power(d)
        powers.append(f"{hw}: {p:.1f}W")
    print(f"  {name}: {', '.join(powers)}")

# Scaling law parameters summary
print("\n─── Scaling Law Parameters (E = α·N^β) ───")
for key, val in scaling_laws.items():
    if not key.startswith("crossover"):
        print(f"  {key}: α={val['alpha']:.4f}, β={val['beta']:.4f}, R²={val['R2']:.4f}")

print("\n" + "="*70)
print("  Analysis complete! All figures and data saved.")
print("="*70)
