#!/usr/bin/env python3
"""Generate Nature Communications source data CSV files for all figures and tables."""

import json
import csv
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(os.path.expanduser("~/Documents/project_energy/results"))
OUTPUT_DIR = Path(os.path.expanduser("~/Documents/project_energy/paper/source_data"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === File loading config ===
FILE_CONFIGS = {
    # (filename, default_platform)
    # Mac experiments (no hardware field in some)
    "exp1_text.json": "Mac M2 Ultra",
    "exp2_image.json": "Mac M2 Ultra",
    "exp3_video.json": "Mac M2 Ultra",
    "exp4_music.json": "Mac M2 Ultra",
    # A100 experiments
    "exp1_text_A100.json": "NVIDIA A100",
    "exp2_image_A100.json": "NVIDIA A100",
    "exp3_video_A100.json": "NVIDIA A100",
    "exp4_music_A100.json": "NVIDIA A100",
    # H100 experiments
    "text_phi3_H100.json": "NVIDIA H100",
    "image_sd15_H100.json": "NVIDIA H100",
    "video_animatediff_H100.json": "NVIDIA H100",
    "image_sdxl_H100.json": "NVIDIA H100",
    "batched_phi3_H100.json": "NVIDIA H100",
    # Extra Mac/A100
    "exp2_image_extra_Mac.json": "Apple M4 Pro",
    "exp2_image_extra_A100.json": "NVIDIA A100",
    "exp3_video_extra_Mac.json": "Apple M4 Pro",
    "exp3_video_extra_A100.json": "NVIDIA A100",
    # SDXL
    "exp5_sdxl_Mac.json": "Apple M4 Pro",
    "exp5_sdxl_A100.json": "NVIDIA A100",
    # Batched
    "exp7_batched_A100.json": "NVIDIA A100",
    # Quality
    "quality_image_sd15_Mac.json": "Apple M4 Pro",
}


def load_all_data():
    """Load all JSON data files and normalize fields."""
    all_records = []
    file_counts = {}
    
    for filename, default_platform in FILE_CONFIGS.items():
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            print(f"WARNING: {filename} not found, skipping")
            continue
        
        with open(filepath) as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"WARNING: {filename} is not a list, skipping")
            continue
        
        valid = 0
        for rec in data:
            # Filter out records missing total_energy_j
            if rec.get("total_energy_j") is None:
                continue
            
            # Normalize platform
            hw = rec.get("hardware", default_platform)
            if "H100" in str(hw):
                platform = "NVIDIA H100"
            elif "A100" in str(hw):
                platform = "NVIDIA A100"
            elif "M4" in str(hw):
                platform = "Apple M4 Pro"
            elif "M2" in str(hw) or default_platform == "Mac M2 Ultra":
                platform = "Mac M2 Ultra"
            else:
                platform = default_platform
            
            # Normalize fields
            normalized = {
                "platform": platform,
                "modality": rec.get("modality", "unknown"),
                "model": rec.get("model", "unknown"),
                "total_energy_j": rec["total_energy_j"],
                "avg_power_w": rec.get("avg_power_w"),
                "peak_power_w": rec.get("max_power_w"),
                "generation_time_s": rec.get("generation_time_sec"),
                "run": rec.get("run", 1),
                "source_file": filename,
            }
            
            # Build configuration string based on modality
            modality = rec.get("modality", "")
            if modality == "text":
                normalized["configuration"] = f"max_tokens={rec.get('max_tokens', 'N/A')}"
                normalized["params_B"] = rec.get("params_B")
                normalized["architecture"] = "Transformer (decoder)"
            elif modality == "image":
                normalized["configuration"] = f"{rec.get('resolution','N/A')}_steps={rec.get('steps','N/A')}"
                normalized["params_B"] = rec.get("params_B")
                normalized["architecture"] = "U-Net (diffusion)"
            elif modality == "video":
                normalized["configuration"] = f"{rec.get('resolution','N/A')}_frames={rec.get('frames','N/A')}_steps={rec.get('steps','N/A')}"
                normalized["params_B"] = None  # AnimateDiff doesn't have params_B in data
                normalized["architecture"] = "U-Net + motion (diffusion)"
            elif modality == "music":
                params_m = rec.get("params_M")
                params_b = params_m / 1000.0 if params_m else None
                normalized["configuration"] = f"tokens={rec.get('max_tokens','N/A')}_audio={rec.get('audio_sec','N/A')}s"
                normalized["params_B"] = params_b
                normalized["architecture"] = "Transformer (encoder-decoder)"
            elif modality == "text_batched":
                normalized["configuration"] = f"batch={rec.get('batch_size','N/A')}_tokens={rec.get('max_tokens','N/A')}"
                normalized["params_B"] = rec.get("params_B")
                normalized["architecture"] = "Transformer (decoder)"
                normalized["per_query_energy_j"] = rec.get("per_query_energy_j")
            else:
                normalized["configuration"] = "default"
                normalized["params_B"] = rec.get("params_B")
                normalized["architecture"] = "unknown"
            
            # Extras
            normalized["clip_score"] = rec.get("clip_score")
            normalized["batch_size"] = rec.get("batch_size")
            normalized["resolution"] = rec.get("resolution")
            normalized["steps"] = rec.get("steps")
            normalized["frames"] = rec.get("frames")
            normalized["max_tokens"] = rec.get("max_tokens")
            normalized["audio_sec"] = rec.get("audio_sec")
            
            all_records.append(normalized)
            valid += 1
        
        file_counts[filename] = valid
    
    print(f"\nLoaded {len(all_records)} total valid records from {len(file_counts)} files:")
    for fname, count in sorted(file_counts.items()):
        print(f"  {fname}: {count} records")
    
    return all_records


def write_csv(filepath, rows, fieldnames):
    """Write a CSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Written: {filepath.name} ({len(rows)} rows)")


def generate_all_raw_data(records):
    """Master file with ALL individual measurements."""
    fieldnames = [
        "platform", "modality", "model", "configuration",
        "run_number", "total_energy_j", "avg_power_w", "peak_power_w",
        "generation_time_s"
    ]
    rows = []
    for r in records:
        rows.append({
            "platform": r["platform"],
            "modality": r["modality"],
            "model": r["model"],
            "configuration": r["configuration"],
            "run_number": r["run"],
            "total_energy_j": round(r["total_energy_j"], 4) if r["total_energy_j"] is not None else "",
            "avg_power_w": round(r["avg_power_w"], 4) if r["avg_power_w"] is not None else "",
            "peak_power_w": round(r["peak_power_w"], 4) if r["peak_power_w"] is not None else "",
            "generation_time_s": round(r["generation_time_s"], 4) if r["generation_time_s"] is not None else "",
        })
    
    write_csv(OUTPUT_DIR / "all_raw_data.csv", rows, fieldnames)
    return len(rows)


def generate_fig1(records):
    """Fig 1: 4-modal comparison. Mean/std energy by platform × modality × model × configuration."""
    # Group by (platform, modality, model, configuration)
    groups = defaultdict(list)
    # Only include the core 4 modalities
    core_modalities = {"text", "image", "video", "music"}
    for r in records:
        if r["modality"] in core_modalities:
            key = (r["platform"], r["modality"], r["model"], r["configuration"])
            groups[key].append(r["total_energy_j"])
    
    fieldnames = ["platform", "modality", "model", "configuration", "mean_energy_j", "std_energy_j", "n_runs"]
    rows = []
    for (platform, modality, model, config), energies in sorted(groups.items()):
        rows.append({
            "platform": platform,
            "modality": modality,
            "model": model,
            "configuration": config,
            "mean_energy_j": round(np.mean(energies), 4),
            "std_energy_j": round(np.std(energies, ddof=1), 4) if len(energies) > 1 else 0.0,
            "n_runs": len(energies),
        })
    
    write_csv(OUTPUT_DIR / "fig1_source_data.csv", rows, fieldnames)
    return len(rows)


def generate_fig2(records):
    """Fig 2: Efficiency ratio — GPU energy vs Mac energy for matching configurations."""
    # Build lookup: (modality, model, configuration) → {platform: mean_energy}
    core_modalities = {"text", "image", "video", "music"}
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["modality"] in core_modalities:
            key = (r["modality"], r["model"], r["configuration"])
            groups[key][r["platform"]].append(r["total_energy_j"])
    
    fieldnames = ["platform_pair", "modality", "configuration", "gpu_energy_j", "mac_energy_j", "ratio"]
    rows = []
    
    mac_platforms = ["Mac M2 Ultra", "Apple M4 Pro"]
    gpu_platforms = ["NVIDIA A100", "NVIDIA H100"]
    
    for (modality, model, config), platform_data in sorted(groups.items()):
        for mac_p in mac_platforms:
            if mac_p not in platform_data:
                continue
            mac_mean = np.mean(platform_data[mac_p])
            for gpu_p in gpu_platforms:
                if gpu_p not in platform_data:
                    continue
                gpu_mean = np.mean(platform_data[gpu_p])
                ratio = gpu_mean / mac_mean if mac_mean > 0 else None
                rows.append({
                    "platform_pair": f"{gpu_p} vs {mac_p}",
                    "modality": modality,
                    "configuration": f"{model}_{config}",
                    "gpu_energy_j": round(gpu_mean, 4),
                    "mac_energy_j": round(mac_mean, 4),
                    "ratio": round(ratio, 4) if ratio is not None else "",
                })
    
    write_csv(OUTPUT_DIR / "fig2_source_data.csv", rows, fieldnames)
    return len(rows)


def generate_fig3(records):
    """Fig 3: Speed-energy tradeoff. Normalized to smallest config per platform×modality."""
    core_modalities = {"text", "image", "video", "music"}
    
    # Group by (platform, modality, model)
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["modality"] in core_modalities:
            key = (r["platform"], r["modality"], r["model"])
            groups[key][r["configuration"]].append(r)
    
    fieldnames = ["platform", "modality", "configuration", "speed_ratio", "energy_ratio"]
    rows = []
    
    for (platform, modality, model), config_data in sorted(groups.items()):
        # Find baseline (smallest config = first sorted config, typically smallest tokens/resolution)
        sorted_configs = sorted(config_data.keys())
        baseline_config = sorted_configs[0]
        baseline_records = config_data[baseline_config]
        
        base_time = np.mean([r["generation_time_s"] for r in baseline_records if r["generation_time_s"]])
        base_energy = np.mean([r["total_energy_j"] for r in baseline_records])
        
        if base_time == 0 or base_energy == 0:
            continue
        
        for config in sorted_configs:
            recs = config_data[config]
            mean_time = np.mean([r["generation_time_s"] for r in recs if r["generation_time_s"]])
            mean_energy = np.mean([r["total_energy_j"] for r in recs])
            
            rows.append({
                "platform": platform,
                "modality": f"{modality} ({model})",
                "configuration": config,
                "speed_ratio": round(mean_time / base_time, 4) if base_time > 0 else "",
                "energy_ratio": round(mean_energy / base_energy, 4) if base_energy > 0 else "",
            })
    
    write_csv(OUTPUT_DIR / "fig3_source_data.csv", rows, fieldnames)
    return len(rows)


def generate_fig4(records):
    """Fig 4: Power profile — avg and peak power per platform×modality×configuration."""
    groups = defaultdict(list)
    core_modalities = {"text", "image", "video", "music"}
    for r in records:
        if r["modality"] in core_modalities:
            key = (r["platform"], r["modality"], r["model"], r["configuration"])
            groups[key].append(r)
    
    fieldnames = ["platform", "modality", "configuration", "avg_power_w", "peak_power_w"]
    rows = []
    
    for (platform, modality, model, config), recs in sorted(groups.items()):
        avg_powers = [r["avg_power_w"] for r in recs if r["avg_power_w"] is not None]
        peak_powers = [r["peak_power_w"] for r in recs if r["peak_power_w"] is not None]
        
        rows.append({
            "platform": platform,
            "modality": f"{modality} ({model})",
            "configuration": config,
            "avg_power_w": round(np.mean(avg_powers), 4) if avg_powers else "",
            "peak_power_w": round(np.mean(peak_powers), 4) if peak_powers else "",
        })
    
    write_csv(OUTPUT_DIR / "fig4_source_data.csv", rows, fieldnames)
    return len(rows)


def generate_table1(records):
    """Table 1: Scaling law parameters — fit energy = alpha * complexity^beta per modality×platform."""
    core_modalities = {"text", "image", "video", "music"}
    
    # Group by (modality, model, platform)
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["modality"] in core_modalities:
            key = (r["modality"], r["model"], r["platform"])
            # Extract complexity proxy
            complexity = _extract_complexity(r)
            if complexity is not None and complexity > 0:
                groups[key][complexity].append(r["total_energy_j"])
    
    fieldnames = ["modality", "platform", "alpha", "beta", "r_squared"]
    rows = []
    
    for (modality, model, platform), complexity_data in sorted(groups.items()):
        if len(complexity_data) < 3:
            continue
        
        # Get mean energy per complexity level
        x_vals = []
        y_vals = []
        for complexity, energies in sorted(complexity_data.items()):
            x_vals.append(complexity)
            y_vals.append(np.mean(energies))
        
        x = np.array(x_vals)
        y = np.array(y_vals)
        
        # Fit log-log: log(y) = log(alpha) + beta * log(x)
        try:
            log_x = np.log(x)
            log_y = np.log(y)
            
            # Linear regression in log space
            n = len(log_x)
            sum_lx = np.sum(log_x)
            sum_ly = np.sum(log_y)
            sum_lx2 = np.sum(log_x**2)
            sum_lxly = np.sum(log_x * log_y)
            
            beta = (n * sum_lxly - sum_lx * sum_ly) / (n * sum_lx2 - sum_lx**2)
            log_alpha = (sum_ly - beta * sum_lx) / n
            alpha = np.exp(log_alpha)
            
            # R-squared
            ss_res = np.sum((log_y - (log_alpha + beta * log_x))**2)
            ss_tot = np.sum((log_y - np.mean(log_y))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            rows.append({
                "modality": f"{modality} ({model})",
                "platform": platform,
                "alpha": round(alpha, 6),
                "beta": round(beta, 4),
                "r_squared": round(r_squared, 4),
            })
        except Exception as e:
            print(f"  Warning: Could not fit scaling law for {modality}/{model}/{platform}: {e}")
    
    write_csv(OUTPUT_DIR / "table1_source_data.csv", rows, fieldnames)
    return len(rows)


def _extract_complexity(r):
    """Extract a complexity proxy from a record based on modality."""
    modality = r["modality"]
    if modality == "text":
        return r.get("max_tokens")
    elif modality == "image":
        res = r.get("resolution", "")
        steps = r.get("steps")
        if "x" in str(res) and steps:
            w = int(res.split("x")[0])
            return w * w * steps  # pixels × steps
        return None
    elif modality == "video":
        res = r.get("resolution", "")
        steps = r.get("steps")
        frames = r.get("frames")
        if "x" in str(res) and steps and frames:
            w = int(res.split("x")[0])
            return w * w * steps * frames
        return None
    elif modality == "music":
        return r.get("max_tokens")
    return None


def generate_table2(records):
    """Table 2: Reliability/CV — coefficient of variation per modality×configuration×platform."""
    groups = defaultdict(list)
    core_modalities = {"text", "image", "video", "music"}
    for r in records:
        if r["modality"] in core_modalities:
            key = (r["modality"], r["configuration"], r["platform"])
            groups[key].append(r["total_energy_j"])
    
    fieldnames = ["modality", "configuration", "platform", "cv_percent", "n_runs"]
    rows = []
    
    for (modality, config, platform), energies in sorted(groups.items()):
        if len(energies) < 2:
            continue
        mean_e = np.mean(energies)
        std_e = np.std(energies, ddof=1)
        cv = (std_e / mean_e) * 100 if mean_e > 0 else 0
        
        rows.append({
            "modality": modality,
            "configuration": config,
            "platform": platform,
            "cv_percent": round(cv, 4),
            "n_runs": len(energies),
        })
    
    write_csv(OUTPUT_DIR / "table2_source_data.csv", rows, fieldnames)
    return len(rows)


def generate_table3(records):
    """Table 3: Model configurations summary."""
    # Gather unique (modality, model) combinations
    model_info = defaultdict(lambda: {
        "architecture": "unknown",
        "params": None,
        "platforms": set(),
        "configs": set(),
    })
    
    for r in records:
        key = (r["modality"], r["model"])
        info = model_info[key]
        info["architecture"] = r.get("architecture", "unknown")
        if r.get("params_B") is not None:
            info["params"] = r["params_B"]
        info["platforms"].add(r["platform"])
        info["configs"].add(r["configuration"])
    
    fieldnames = ["modality", "model", "architecture", "params", "configs_per_platform", "platforms"]
    rows = []
    
    for (modality, model), info in sorted(model_info.items()):
        n_platforms = len(info["platforms"])
        n_configs = len(info["configs"])
        params_str = f"{info['params']}B" if info["params"] is not None else "N/A"
        
        rows.append({
            "modality": modality,
            "model": model,
            "architecture": info["architecture"],
            "params": params_str,
            "configs_per_platform": n_configs,
            "platforms": "; ".join(sorted(info["platforms"])),
        })
    
    write_csv(OUTPUT_DIR / "table3_source_data.csv", rows, fieldnames)
    return len(rows)


def main():
    print("=" * 60)
    print("Nature Communications Source Data Generator")
    print("=" * 60)
    
    records = load_all_data()
    
    print(f"\n{'=' * 60}")
    print("Generating source data CSV files...")
    print("=" * 60)
    
    summary = {}
    summary["all_raw_data.csv"] = generate_all_raw_data(records)
    summary["fig1_source_data.csv"] = generate_fig1(records)
    summary["fig2_source_data.csv"] = generate_fig2(records)
    summary["fig3_source_data.csv"] = generate_fig3(records)
    summary["fig4_source_data.csv"] = generate_fig4(records)
    summary["table1_source_data.csv"] = generate_table1(records)
    summary["table2_source_data.csv"] = generate_table2(records)
    summary["table3_source_data.csv"] = generate_table3(records)
    
    print(f"\n{'=' * 60}")
    print("SUMMARY — Records per file:")
    print("=" * 60)
    total = 0
    for fname, count in summary.items():
        print(f"  {fname:30s} → {count:6d} rows")
        total += count
    print(f"  {'TOTAL':30s} → {total:6d} rows")
    print(f"\nAll files written to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
