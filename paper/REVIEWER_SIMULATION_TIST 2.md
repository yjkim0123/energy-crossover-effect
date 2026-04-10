# Comprehensive Adversarial Review Simulation
## "The Energy Crossover Effect: How Model Architecture Shapes Hardware Efficiency for Generative AI Deployment"
### Target: ACM Transactions on Intelligent Systems and Technology (TIST)

**Date:** 2026-02-26  
**Verification Basis:** Full paper (`main_tist.tex`), 4,703 raw JSON records, `verify_paper_numbers.py` output, `cross_hardware_analysis.py`, all `scripts/`, and web-based literature review.

---

## Pre-Review: Data Verification Summary

The verification script (`verify_paper_numbers.py`) found **57/59 matches** with 2 flagged mismatches. Our extended manual verification uncovered several **additional issues** the script missed:

### Issues the Script Caught ✅
- `scaling_laws_3hw.json` is stale (wrong fitting variables)
- `analysis_stats.json` reports 1,020 runs (stale; actual: 4,703)
- H100 image R² corrected from 0.977 → 0.877 in TIST version

### Issues the Script MISSED 🔴🟡

| # | Severity | Issue | Paper Claim | Actual Data |
|---|----------|-------|-------------|-------------|
| 1 | 🔴 | **Video 16f energy saving** | "64% at 16 frames" (§3.1) | **45%** (1027→565J, saving=45.0%) |
| 2 | 🔴 | **A100 video bimodal data** | Treated as unimodal, 30 runs | 60 runs with bimodal distribution (clusters at ~335J and ~1015J for 4f); CV ≫ reported |
| 3 | 🔴 | **PyTorch version** | "PyTorch 2.10.0" (§2.1) | PyTorch 2.10.0 does not exist. Likely typo for 2.1.0 |
| 4 | 🟡 | **H100 power "~1.3×"** | "draws ~1.3× more power than A100" (§3.5, Fig 8 caption) | **2.0–2.8×** for image; only ~1.4–2.0× for video |
| 5 | 🟡 | **H100 384² speedup "~22×"** | "speedup ~22×" (§3.5) | **14.9×** (6.72s→0.45s duration_sec) |
| 6 | 🟡 | **H100 384² power ratio "~10×"** | "power ratio ~10×" (§3.5) | **14.5×** (17W→251W) |
| 7 | 🟡 | **A100 "13.2× faster"** | "A100 completes 512² generation 13× faster" (§4.4) | 6.7× by duration_sec; 13.2× by generation_time_sec. Energy uses duration, speedup uses gen_time—inconsistent |
| 8 | 🟡 | **Batched comparison** | "69J remains 2.8× higher than Mac's 24.5J at 64 tokens" (§3.7) | Batched uses 256 tokens; Mac at 256tok = 127.5J. H100@batch=16 (69J) actually **beats** Mac at same token count |
| 9 | 🟡 | **Music power ratio** | "power ratio 9.9×" (§3.5) | **7.7×** (A100 71W / Mac 9.2W) |
| 10 | 🟡 | **H100 "412W at 640²"** | "412W at 640²" (§3.2) | Mean across all steps: 378W. Only 640²/50-step avg is 412W (cherry-picked) |
| 11 | 🟡 | **H100 vs A100 image savings** | "35–45%" (§3.3) | Range: −20% to 42% (includes 128² where H100 is *worse*) |
| 12 | 🟡 | **H100 vs A100 video savings** | "30–65%" (§3.3) | Range: 19–69% (wider than claimed) |
| 13 | 🟡 | **Prompt diversity text range** | "11.0%" (§6) | **8.4%** from data |
| 14 | 🟡 | **Image within/between variance** | "13×" (§6) | **7.6×** from data |
| 15 | 🟡 | **Reference [23]** | Xu & Jia, "Optimizing edge AI" | arXiv:2501.03265 is actually "Cognitive Edge Computing" by different authors |
| 16 | 🟡 | **Reference [1] year** | bibkey `iea2024` | Text body says "IEA, Paris, 2025" |

---

## R1: Green AI / Sustainable Computing Expert

### Summary
This paper benchmarks energy consumption of generative AI across three hardware platforms (edge SoC, A100, H100) and four modalities, identifying an "Energy Crossover Effect" where the optimal hardware shifts as task complexity increases. The cross-hardware, cross-modality approach fills a gap in the Green AI literature, which has been predominantly GPU-centric.

### Strengths
1. **Addresses a genuine blind spot in Green AI.** Prior work (Strubell et al., Patterson et al., Luccioni et al.) has focused on training or single-hardware inference. This is the first systematic study showing that hardware *selection* can yield order-of-magnitude savings—a finding with direct policy relevance.
2. **Actionable deployment guidelines.** The routing framework (autoregressive → edge, diffusion → GPU above crossover) is immediately implementable. The crossover thresholds provide concrete decision boundaries.
3. **Comprehensive sensitivity analysis.** The 40% PUE correction addresses the most common criticism of software-based power measurement, and the prompt invariance experiment demonstrates energy is configuration-determined rather than content-determined.
4. **Strong data volume.** 4,500+ runs with 30 repeats per configuration provides solid statistical grounding for the main claims.

### Weaknesses
1. 🔴 **Software power measurement is fundamentally limited.** The Mac uses `powermetrics` (all subsystems, hardware counters) while GPUs use `pynvml` (board power only). This asymmetry biases every comparison. While the paper acknowledges this and applies a 40% correction, it does not discuss:
   - The correction should be applied *per-modality* (GPU utilization varies dramatically, so PUE analog varies)
   - `powermetrics` includes DRAM power but `pynvml` excludes host DRAM—significant for text generation where the model is memory-bound
   - No external power meter validation (e.g., wall-socket measurement) for either platform

2. 🔴 **No carbon or lifecycle analysis despite framing.** The paper opens with IEA projections and climate framing but never calculates carbon emissions, embodied energy, or manufacturing footprint. An edge device operating 24/7 versus a shared GPU serving thousands of users has fundamentally different amortization profiles. The claim that routing to edge "reduces energy" ignores that a GPU serving batch=16 (69J/query) already approaches edge efficiency, and the GPU's energy is *shared* across tenants.

3. 🟡 **Missing comparison with CodeCarbon/CarbonTracker.** The paper cites Henderson et al. [24] and Lacoste et al. [25] but does not compare its measurement methodology against these established tools. Given that CodeCarbon is the de facto standard for ML energy reporting, this is an oversight.

4. 🟡 **The routing benefit estimate conflates single-tenant edge with multi-tenant GPU.** "Up to 10× savings for autoregressive" assumes dedicated GPU per query. In practice, GPUs serve batched queries; at batch=16, the H100 per-query energy (69J) for 256-token text is already *lower* than Mac (127.5J)—the opposite of the paper's narrative.

### Questions for Authors
1. Have you validated `powermetrics` readings against wall-socket power measurements? What is the measurement error bound?
2. How would the crossover analysis change if you accounted for the number of concurrent users a GPU can serve versus a single-user edge device?
3. Could you compute and report CO₂-equivalent emissions using regional grid intensities (e.g., California vs. coal-heavy grids)?
4. What is the embodied energy of the Mac mini vs. a share of an H100 server? Does this change the lifecycle conclusion?

### Verdict: Minor Revision
### Confidence: 4/5

---

## R2: Systems/Architecture Expert

### Summary
The paper compares energy consumption of generative AI inference on an Apple M4 Pro edge SoC, NVIDIA A100, and H100 GPUs across four modalities. It reports that autoregressive tasks always favor edge while diffusion tasks exhibit a GPU-favorable crossover at higher complexity. The experimental methodology uses software-level power monitoring with 30 runs per configuration.

### Strengths
1. **Three-tier hardware comparison is valuable.** Most benchmarks are single-platform. Including the M4 Pro, A100, and H100 enables analysis of generational trends (A100 → H100) alongside the edge-cloud dimension.
2. **The crossover mechanism explanation is physically sound.** The E = P·t decomposition showing that GPUs win when speedup > power ratio is intuitive and backed by data.
3. **Batched inference experiments bridge the gap to real deployment.** Including batch sizes 1–16 for both A100 and H100 adds practical value.
4. **30 runs per configuration with CV reporting.** Good experimental practice.

### Weaknesses
1. 🔴 **A100 video data has bimodal distribution.** For 4-frame and 8-frame A100 video, the 60 data points cluster in two distinct groups (~335J and ~1015J for 4f; ~470J and ~1770J for 8f). This is classic Colab multi-tenancy behavior—the GPU is either throttled or shared. Using the mean of a bimodal distribution is statistically meaningless. This affects *all* A100 video comparisons and the reported crossover at ~7 frames.

2. 🔴 **PyTorch version mismatch.** The paper states "PyTorch 2.10.0" on GPUs, but this version does not exist (PyTorch uses semantic versioning 2.x.y where x < 10 as of Feb 2026). This calls into question what software was actually running. If it should be 2.1.0, that's 3 minor versions behind the Mac (2.4.1)—a significant gap that could affect performance comparisons, particularly for MPS vs CUDA backends.

3. 🔴 **Inconsistent time metrics inflate speedup claims.** The paper cites "A100 completes 512² generation 13× faster" using `generation_time_sec` (pure model compute), but energy is calculated over `duration_sec` (includes pipeline overhead). The actual speedup by `duration_sec` is 6.7×. This inconsistency affects the §4.4 mechanism analysis: the actual speedup (6.7×) barely exceeds the power ratio (5.9×), making the claimed "energy advantage" much more marginal than presented.

4. 🟡 **No thermal throttling analysis.** The Mac mini operates passively cooled in a small enclosure. Sustained workloads will trigger thermal throttling, affecting the 30-run measurements. No temperature data is reported. For the Mac at 512² image (13s × 30 = 6.5 min sustained), throttling is likely.

5. 🟡 **Power measurement granularity differs.** `powermetrics` at 100ms and `pynvml` at 50ms create different integration errors for short tasks. For H100 image generation at 128² (0.15s duration), only ~3 power samples are captured—far too few for accurate energy estimation.

### Questions for Authors
1. What was the actual PyTorch version on GPUs? Is "2.10.0" a typo for "2.1.0"?
2. For the A100 video bimodal data: did you investigate the cause? Were these from different Colab sessions? Why were both populations included in the analysis?
3. Did you monitor CPU/SoC temperature during Mac experiments? What was the ambient temperature?
4. For H100 measurements lasting <0.5s, how many power samples were captured? What is the integration error?
5. Why use `generation_time_sec` for speedup claims but `duration_sec` for energy? Can you reconcile this?

### Verdict: Major Revision
### Confidence: 5/5

---

## R3: ML Benchmarking Expert

### Summary
The paper presents a cross-hardware, cross-modality energy benchmark with 6 models, 76 configurations, and 4,500+ runs. It identifies modality-dependent hardware efficiency patterns and proposes energy-aware inference routing guidelines.

### Strengths
1. **Multi-modality coverage is novel.** Text, image, video, and music in a single benchmark enables the modality-dependent crossover finding that wouldn't emerge from single-modality studies.
2. **Reproducible setup.** Open-weight models, standard frameworks (PyTorch + Hugging Face), and planned data release via GitHub enable replication.
3. **Cross-model validation with SDXL.** Testing a second image model (SDXL, 3.5B) confirms patterns aren't model-specific.
4. **Energy-quality analysis.** CLIP scores alongside energy data enables Pareto frontier analysis, which is rare in energy benchmarks.

### Weaknesses
1. 🔴 **Model selection is too narrow for "comprehensive" claims.** 6 models is modest by current standards:
   - **No DiT-based models** (Stable Diffusion 3, Flux)—the dominant architecture for 2025 image generation. UNet-based SD v1.5 is effectively deprecated.
   - **No LLMs >3.8B.** Phi-3-mini is an edge-optimized model; testing Llama 3 (8B, 70B) or Mistral would show how the crossover shifts with model size.
   - **Music has only 2 models** (MusicGen small/medium) and was tested on only 2 platforms (no H100).
   - The paper acknowledges this but the limitation significantly weakens the generalizability claim.

2. 🟡 **Configuration space coverage is unbalanced.** H100 image has 30 configs (6 resolutions × 6 step counts) while Mac/A100 image has only 6. The scaling law fits therefore have different statistical power across platforms. The H100 R² = 0.877 (poor) vs Mac R² = 0.9998 (excellent) partly reflects this asymmetry.

3. 🟡 **No comparison with MLPerf Inference or other standard benchmarks.** MLPerf has a well-defined inference benchmark suite including image generation (Stable Diffusion). Positioning relative to this benchmark would strengthen the systems contribution.

4. 🟡 **Text generation uses fixed prompts with max_tokens.** Real LLM inference has highly variable output lengths. The high CV for text (17–19%) likely reflects early EOS tokens, but the paper uses max_tokens as the scaling variable rather than actual tokens generated. This conflates the intended length with the actual computation.

5. 🟡 **Prompt diversity experiment is Mac-only.** Prompt invariance should be validated across all platforms. A GPU's high idle power might create different sensitivity patterns.

### Questions for Authors
1. Would you predict the same crossover pattern for DiT-based architectures (SD3, Flux), which have different compute profiles than UNet?
2. Have you tested or estimated how the crossover shifts for larger LLMs (e.g., 8B, 13B)?
3. How do actual tokens generated (vs. max_tokens) affect the text energy scaling law?
4. Why was MusicGen not tested on the H100?
5. Could you map your findings to MLPerf Inference benchmarks for comparability?

### Verdict: Minor Revision
### Confidence: 4/5

---

## R4: Statistician

### Summary
The paper fits power-law scaling models to energy-complexity relationships across hardware platforms and uses these to derive crossover points. The statistical analysis includes 30 replications per configuration, Welch's t-tests, Cohen's d, and R² for regression fits.

### Strengths
1. **Appropriate use of nonlinear least squares (Levenberg-Marquardt).** Fitting E = α·N^β directly rather than OLS on log-log avoids heteroscedasticity bias from the log transformation.
2. **30 replicates per config is adequate.** With 30 runs, the standard error of the mean is ~σ/5.5, giving reasonably precise estimates.
3. **Strong fits for most combinations.** R² > 0.99 for 10/12 modality-hardware pairs, with clear explanations for the two exceptions (H100 image R² = 0.877; music R² lower due to stochastic sampling).
4. **Cohen's d > 5 for key comparisons.** Effect sizes are enormous, making the main findings robust to measurement noise.

### Weaknesses
1. 🔴 **No confidence intervals on scaling parameters or crossover points.** The crossover point N* = (α_edge/α_GPU)^(1/(β_GPU − β_edge)) is a function of four estimated parameters. Without propagating uncertainties, the paper cannot state whether the image crossover at "~217,000 pixels" has a 95% CI of [200k, 240k] or [100k, 500k]. This is critical for the practical routing guidelines.

2. 🔴 **Crossover computed from fitted parameters, not validated empirically.** The paper claims image crossover at ~466² for A100, but the actual empirical data confirms crossover between 384² (147k pixels) and 512² (262k pixels). The formula gives 217k, which is in this range but the wide bracket shows the formula-based estimate isn't precise. For video, the formula-based crossover for Mac-A100 at "~7 frames" cannot be empirically confirmed because the A100 data is bimodal.

3. 🟡 **R² alone is insufficient for model adequacy.** Residual analysis (heteroscedasticity tests, normality of residuals) should accompany R². With only 3–5 data points per fit (3 resolutions for Mac/A100 image, 5 for H100 image), model overfitting is a concern. A power law with 2 parameters fit to 3 points will always look good.

4. 🟡 **No model comparison.** Is E = α·N^β better than E = α·N + c (linear + intercept) or E = a·N² + b·N + c? For several modality-hardware combinations, β ≈ 1 (near-linear), so the power law is approximately linear. The paper should show the power law is the *best* model, not just that it fits.

5. 🟡 **Multiple testing.** The paper presents ~60 verification checks and multiple hypothesis tests without correction for multiple comparisons. While the effect sizes are large enough that this likely doesn't matter, it should be acknowledged.

### Questions for Authors
1. Can you provide 95% confidence intervals for all α, β estimates and for the crossover points?
2. With only 3 data points for Mac image scaling (3 resolutions), the fit is underdetermined—have you considered using cross-validation or bootstrap?
3. Have you tested the power law against alternative functional forms (e.g., linear, quadratic, E = α·N^β + c)?
4. The supplementary shows confidence bands on scaling fits—why are these not in the main paper?
5. For the A100 video data with bimodal distribution, have you checked whether the scaling law holds *within* each mode separately?

### Verdict: Minor Revision
### Confidence: 5/5

---

## R5: Associate Editor — ACM TIST Fit, Novelty, Impact

### Summary
The paper addresses the energy efficiency of generative AI inference across edge and cloud hardware, identifying a modality-dependent crossover effect. It targets ACM TIST (IF 7.2), which publishes work on intelligent systems spanning AI, systems, and applications. The cross-hardware, cross-modality benchmark fills a gap in the literature and provides actionable deployment guidelines.

### Strengths
1. **Strong TIST fit.** The paper bridges AI models, systems architecture, and energy-efficient computing—an interdisciplinary topic central to TIST's scope. The deployment guidelines add practical value beyond empirical observation.
2. **Timely and significant.** With AI inference projected to dominate data center energy, understanding when edge hardware is more efficient than cloud GPUs is immediately relevant to industry practitioners and policymakers.
3. **Clear, well-structured writing.** The paper is well-organized, progressing logically from experimental setup → results → implications. Figures are clear and informative.
4. **Novel conceptual contribution.** The "Energy Crossover Effect" is a clean, memorable concept. The connection to scaling law exponents (superlinear edge vs. sublinear GPU) provides theoretical grounding.

### Weaknesses
1. 🔴 **Several numerical claims do not match the data.** The "64% at 16 frames" video saving is actually 45%. The H100/A100 power ratio is not "~1.3×" but 2–2.8× for images. The batched comparison (69J vs 24.5J) mixes different token counts. These errors erode trust in the entire quantitative narrative and must be corrected.

2. 🟡 **Insufficient novelty for a top journal?** The paper is essentially an empirical benchmark—well-executed but methodologically straightforward. The power-law fitting extends Kaplan et al. to inference energy, but the approach is standard curve fitting. Iyengar et al. (2025) independently arrived at FLOPs-based energy scaling laws for diffusion. The contribution may be more suited to a systems conference (e.g., SIGMETRICS, MLSys) than a journal.

3. 🟡 **Single author, rapid data collection.** All data was collected in approximately 2 days (Feb 19–21, 2026). The GPU experiments were on Google Colab Pro+, a shared multi-tenant environment. Sustained benchmarking on Colab is known to produce variable results due to VM migration, GPU sharing, and thermal management differences. This is evident in the bimodal A100 video data.

4. 🟡 **Reference quality.** Multiple reference errors: [23] cites wrong paper for the given arXiv ID; [1] has year mismatch; [18] bibkey year doesn't match publication year; [22] bibkey suggests different first author than listed. These suggest insufficient verification.

5. 🟡 **Limited discussion of practical feasibility.** The routing framework assumes: (a) pre-classification of requests by modality and complexity, (b) availability of both edge and cloud hardware, (c) latency tolerance for edge processing. These assumptions need more discussion. In particular, the paper does not address the network latency and energy overhead of routing decisions themselves.

### Questions for Authors
1. Can you correct the numerical errors identified in the verification (particularly the 64% → 45% video claim)?
2. How do you position this paper relative to Iyengar et al. (2025), which also reports energy scaling laws for diffusion models?
3. What is the replication plan? Will you provide a Docker container or Colab notebook for reproduction?
4. Have you considered dedicated GPU servers (not Colab) for a cleaner comparison?
5. Could you expand the discussion of deployment constraints (network latency, load balancing, privacy) in the routing framework?

### Verdict: Minor Revision
### Confidence: 4/5

---

## Meta-Review Synthesis

### Consensus
All five reviewers recognize the paper's **core contribution**—the Energy Crossover Effect—as novel and timely. The cross-hardware, cross-modality design is valued, and the practical routing guidelines are seen as impactful. However, all reviewers identify **numerical accuracy issues** that must be corrected before publication.

### Key Points of Agreement
- ✅ The crossover concept is valid and well-demonstrated
- ✅ 4,500+ runs with 30 repeats provides adequate statistical power
- ✅ The paper fills a genuine gap: cross-hardware energy comparisons
- ❌ Several numerical claims do not match raw data
- ❌ Measurement methodology has unaddressed asymmetries
- ❌ A100 video data quality is compromised (bimodal distribution)

### Key Points of Disagreement
- **Novelty:** R5 questions whether an empirical benchmark is sufficient for TIST; other reviewers see the scaling law framework as a meaningful theoretical contribution.
- **Measurement quality:** R2 views the Colab-based GPU measurement as a major concern; R1 considers the 40% sensitivity analysis adequate.
- **Model scope:** R3 argues 6 models is too few; other reviewers accept the scope as justified by the edge-viable constraint.

### Overall Assessment
The paper presents a valuable empirical contribution with a clear conceptual finding. However, the numerical errors, data quality issues (A100 video bimodal), and measurement methodology gaps require revision before acceptance. The core finding (crossover effect) is robust and would survive corrections.

---

## Action Items

### 🔴 Must-Fix (Pre-Acceptance)

1. **Correct "64% at 16 frames"** → actual saving is 45%. This appears in §3.1 and potentially in the abstract's "46%" claim. Verify every percentage claim against raw data.

2. **Address A100 video bimodal distribution.** Either:
   - Report median instead of mean, or
   - Separate the two populations and analyze each, or
   - Exclude the anomalous cluster with justification, or
   - Re-run on dedicated hardware.

3. **Fix PyTorch version.** "2.10.0" → correct to actual version used (likely 2.1.0). This affects reproducibility.

4. **Correct H100/A100 power ratio claims.** "~1.3×" is only valid for video; for images it's 2.0–2.8×. Rewrite §3.5 and Fig 8 caption with per-modality ratios.

5. **Fix the batched inference comparison.** The 69J (256-token batched) vs 24.5J (64-token Mac) comparison is misleading. At 256 tokens, the Mac uses 127.5J—the H100 at batch=16 *wins*. Either compare same token counts or explicitly state the comparison crosses token counts.

6. **Reconcile speedup metrics.** Either use `duration_sec` consistently (yielding 6.7× for A100 at 512²) or explain why `generation_time_sec` is appropriate. Currently §4.4 uses generation_time for speedup but duration for energy—the mechanism analysis breaks down.

7. **Fix Reference [23].** arXiv:2501.03265 does not match the cited paper. Find correct URL or replace.

8. **Add confidence intervals on crossover points.** Use bootstrap or delta method to propagate α, β uncertainties to N*.

### 🟡 Should-Fix

9. **Report per-modality H100 vs A100 statistics accurately.** "35–45% image savings" is actually −20% to 42% (including 128²). "30–65% video savings" is actually 19–69%.

10. **Correct prompt diversity numbers.** Text range: 8.4% (not 11.0%). Image within/between variance ratio: 7.6× (not 13×).

11. **Add residual analysis for scaling law fits.** With 3 data points for Mac/A100 image, a 2-parameter model is just interpolation. Acknowledge this limitation or add data points.

12. **Discuss measurement asymmetry more explicitly.** Separate `powermetrics` (includes DRAM, CPU, ANE) vs `pynvml` (GPU board only) into a table showing what each captures.

13. **Fix minor reference issues.** [1] year mismatch (key "2024" but text "2025"); [18] MusicGen year; [22] bibkey author.

14. **Verify the "412W at 640²" claim.** This is the mean of 640²/50-step config specifically, not the overall 640² average (378W). Clarify which.

15. **Correct H100 384² speedup.** Actual: 14.9× (not "~22×"). Actual power ratio: 14.5× (not "~10×").

### 🟢 Nice-to-Have

16. **Add DiT-based model (SD3 or Flux).** Would strengthen generalizability claim for image generation.

17. **Add wall-socket validation for at least one platform.** Even a spot-check would address the measurement asymmetry concern.

18. **Compare against MLPerf Inference.** Position results relative to the standard benchmark.

19. **Test MusicGen on H100.** Completing the 3-platform comparison for all modalities would be cleaner.

20. **Add carbon emission estimates.** With regional grid intensities, the energy data translates directly to CO₂.

21. **Discuss model quantization.** Phi-3 at 4-bit on Mac vs FP16 on GPU would show how quantization interacts with the crossover.

22. **Test at least one larger LLM (8B or 13B).** Shows how model size shifts the crossover.

23. **Include Mistral data.** The `exp6_mistral_Mac.json` and `exp6_mistral_A100.json` files exist but are not discussed in the paper.

---

## Overall Recommendation: **Minor Revision**

The core contribution (Energy Crossover Effect) is novel, timely, and robustly supported by data. However, the paper contains multiple numerical errors that must be corrected, a data quality issue (A100 video bimodal) that needs to be addressed, and measurement methodology gaps that should be discussed more transparently. With corrections to the 🔴 must-fix items, the paper would be suitable for TIST.

**Confidence in recommendation: 4/5**

The main risk is that correcting the numerical errors might weaken some secondary claims (e.g., the video savings at high frame counts). However, the primary finding—modality-dependent crossover driven by divergent scaling exponents—is supported by the data regardless of these corrections.
