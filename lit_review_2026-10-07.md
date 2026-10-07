# Literature check — 2026-10-07 (Stage 3 context)

Three Sonnet research agents searched (A: surrogate error spectra and remedies; B: ML
aeroacoustics and novelty; C: hybrid CAA source errors). The main agent then re-checked
every quote marked **[verified]** against the downloaded PDF text. **[summary]** means the
quote came through a WebFetch summary, not the paper text. **[unverified]** means not
confirmed. Searches were limited (a few dozen queries), so absence of prior work is not
proven.

## 1. Closest framing: MSE neglects low-amplitude content (PDE-Refiner)
- Lippe et al. 2023, *PDE-Refiner*, arXiv:2308.05732 **[verified]**:
  - "neural solvers consistently neglect components of the spatial frequency spectrum that have low amplitude. Although these frequencies have minimal immediate impact, they still impact long-term dynamics"
  - Fig. 1: "The MSE model is only accurate for a small, high-amplitude frequency band".
  - Remedies tested there:
    - pushforward trick: "cannot learn to include the low-amplitude information … as no gradients are backpropagated through the predicted input";
    - Sobolev losses: "using the Sobolev norm losses decrease performance also for FNOs".
  - Fix: multistep denoising refinement.
  - Scope: 1D KS and 2D Kolmogorov flow, with **spatial** spectra only. There is no temporal-frequency or (k, ω) analysis.
- **Relation to us:** our radiating cone is low-amplitude content that is low-k but high-ω. PDE-Refiner's diagnosis is amplitude-based neglect in k. Ours extends it to space–time and attaches a physical observable, the radiated sound. In their case the low-amplitude and high-k modes coincide (KS), so the two causes are confounded. In ours they are separated.

## 2. Spectral bias and "small L2, wrong physics"
- Duraisamy 2026, arXiv:2604.20061 (a perspective, 11 pp.) **[verified]**:
  - "neural surrogates systematically under-resolve high-frequency content due to spectral bias"
  - "A surrogate can achieve excellent L2 error while systematically misrepresenting the physics that matters."
  - It mentions combustor acoustics and thermoacoustic phase sensitivity in passing. It has no sonic-cone or (k, ω) analysis and no measured acoustic error. **Cite it as framing.** It does not pre-empt our measurement.
- Rahaman et al. 2019 (arXiv:1806.08734) and Xu et al. 2019 (arXiv:1901.06523) define spectral bias in input/wavenumber space **[summary]**. Nothing found separates amplitude-based neglect from wavenumber-based bias, or treats low-k/high-ω content.
- McCabe et al. 2023, arXiv:2306.10619 **[verified]**: "The instability in the full spectrum models starts in the high frequencies and grows. This is consistent with aliasing behavior."
  - **Alternative mechanism to rule out:** high-k aliasing as the source of our spurious high-ω content.
- H¹/Sobolev gains are reported elsewhere (Sobolev training for operators, arXiv:2402.09084; Spectral-Refiner, arXiv:2405.17211; arXiv:2210.10890) **[summary]**. Our "H¹ does not help the cone" must therefore be shown empirically. The `scripts/06` weighting argument is analysis, not a trained result.

## 3. Remedies (what each targets)
| remedy | source | targets | expected effect on the cone |
|---|---|---|---|
| pushforward trick | Brandstetter et al. 2022, arXiv:2202.03376 | input distribution shift | none: no gradient through the prediction (PDE-Refiner, verified) |
| temporal bundling | same | fewer calls, less shift | unknown; may change step-to-step coherence [unverified] |
| noise injection | Sanchez-Gonzalez et al. 2020, arXiv:2002.09405 | error accumulation | unknown [summary] |
| Sobolev / H¹ | neuraloperator `H1Loss`; 2210.10890 | high-k | none expected (our `scripts/06`); hurt in PDE-Refiner (verified) |
| multistep refinement | PDE-Refiner | low-amplitude modes | the closest direct fix; heavier to train |
| unrolled loss with backprop through steps | — | temporal coherence | **not found tested for this purpose**; our jitter hypothesis predicts it should lower B |

## 4. Aeroacoustics side
- **The T-form for periodic boxes** is confirmed. Sarkar & Hussaini 1993, ICASE 93-74 **[verified]**: "The surface contribution is an artifact of the boundary conditions and is not of interest to us."
- **Box-boundary artifacts are O(1), not small.** Wang 1994, CTR ARB **[verified]**:
  - "Had the spurious boundary contribution not been subtracted, one would predict incorrectly sound radiation"
  - The boundary flux and the quadrupole source "show oscillations of similar magnitude but opposite phase".
- **Smooth, coherent source errors are benign.** Same brief **[verified]**: "the basic source characteristics are preserved after filtering the flow field with a filter of widths up to 4Δ". This is consistent with our reading that *incoherent* (jitter) error is what radiates.
- **Tapering a source cut-off.** Hybrid FV/SEM, arXiv:2302.03370 **[verified]**: "A smoothing function is employed in order to let the sound source term decay to avoid the well known spurious noise generation due to the abrupt domain cut". This is precedent for our Tukey taper, though theirs is in a non-periodic wake.
- **Sensitivity of vortex sound to non-conservative flow data:** Schram & Hirschberg 2002, JSV 266 **[summary, abstract only]**.
- **2D M⁷ law:** Schoder et al., arXiv:2608.19236 **[verified 2026-10-07, earlier session]**. Ffowcs Williams 1969 itself: **[unverified]**.
- **Rucz et al. 2024**, JSV 587, 118496, "Analysis of the co-rotating vortex pair as a test case for CAA". The metadata is confirmed. Full text was paywalled (403), so whether it gives an absolute amplitude is **[unverified]**. The open-access precursor Ulveczki & Rucz (Forum Acusticum 2023) says the flow and far field "can be determined analytically" **[summary]**.
- A quantitative "source error X → O(1) sound error at M = 0.1" rule (Crighton / Colonius–Lele): **not found [unverified]**.
- Temporal jitter or time interpolation of Lighthill sources as a known spurious-noise mechanism: **not found**.

## 5. Novelty (frank)
- **Not found** (searches limited):
  - surrogate error analysed in (k, ω) against the sonic line;
  - an error-share-in-cone over signal-share ratio (B);
  - measured acoustic error of an ML flow surrogate compared with a trivial baseline;
  - jitter as the mechanism.
- **Established, cite it, don't claim it:** L2 under-weights low-amplitude and high-k content (PDE-Refiner, spectral bias); small L2 error can coexist with wrong physics (Duraisamy).
- **Still to do before claiming priority:** a proper Google Scholar / JFM / JSV / AIAA / PRFluids search by a human, and full reads of PDE-Refiner §4–5 and the Hierarchical Implicit Neural Emulators and Thermalizer papers (arXiv:2506.04528, 2503.18731; not read).
