# The Silent Field

**Neural PDE surrogates lose the acoustic field — but not for the reason the literature assumes.**

Spectral bias says networks learn low wavenumbers first and lose high ones. The
acoustic field is at **low** wavenumber: at Mach number `M` the acoustic
wavelength exceeds the hydrodynamic length scale by `1/M`. By every spectral-bias
argument it should be the easiest content in the flow to learn.

It is destroyed anyway, because it is invisible in **amplitude**, not in frequency.

## The measurement

A disturbance radiates only if its phase speed is supersonic, `ω/|k| ≥ c₀`. That
carves a cone out of the wavenumber–frequency plane. Turbulence does not live
there — it lives on the convective ridge `ω ≈ U|k|`, whose slope is smaller by
exactly the Mach number.

An L2 loss weights every mode by its energy. So the share of the training
objective assigned to the only region that makes sound is:

| Mach | L2 objective | Lighthill source | Lighthill source, T-form (R_T) |
|------|--------------|------------------|--------------------------------|
| 0.05 | 1.8 × 10⁻¹¹  | 5.3 × 10⁻¹² | 7.5 × 10⁻¹¹ |
| 0.10 | 3.2 × 10⁻⁸   | 2.3 × 10⁻⁹  | 4.9 × 10⁻⁹  |
| 0.20 | 4.8 × 10⁻⁶   | 3.4 × 10⁻⁷  | 3.9 × 10⁻⁷  |
| 0.30 | 6.2 × 10⁻⁵   | 5.5 × 10⁻⁶  | 5.9 × 10⁻⁶  |

*(128², L = 2π, `data/turb2d.npz`, Welch 4-segment, full record:
`python scripts/04_komega_analysis.py`. Reproduce in ~3 min on a laptop.)*

The T-form column counts the radiating content below `c₀·dk` (the compact term) that the
box-mode source misses, so it is larger at low Mach (14× at M = 0.05, 1.08× at M = 0.3);
`python scripts/07_tform_metric.py`.

The steep fall of these shares with Mach is a property of the *share*, not of the sound.
The radiated power itself follows the compact 2D quadrupole law: at fixed flow its local
exponent d ln P / d ln M is 3.9–4.3 over M = 0.02–0.4 on all three datasets, i.e.
P ∝ ρU⁷ℓ/c₀⁴, the 2D "M⁷" law when U varies at fixed c₀
(`python scripts/08_mach_scaling.py`).

**Sobolev training — the standard remedy for spectral bias — cannot raise this
number.** At its default weight it does not touch it at all; where it is strong enough to
matter it lowers it. That is the finding.

A Sobolev objective is a *sum* of two separately normalised terms, so the weight it
applies to each full-plane mode is `1 + λ·c·w̃(k)`, not `(1+k²)^order`. Here `w̃` is the
`(1+k²)^order` weight normalised by its own mean, `c = N_cells/N_half` on the rfft2 `kx = 0`
and Nyquist columns, and `c = N_cells/(2N_half)` elsewhere (each interior coefficient stands
for ±k). The weight's mass is at high k, where the cone has no area at all; the cone sits at
low k — it reaches only `k ≤ 6.7`, 1.2% of the resolved `|k|` values. So `w_eff ≥ 1`
everywhere and `|w_eff − 1| ≤ 1.7 × 10⁻⁴` across the entire cone at the default λ. The Sobolev
term moves the cone share by 1.3 × 10⁻⁴ at λ = 10⁻², about 0.01% of the objective:

| λ | 10⁻³ | 10⁻² (default) | 10⁻¹ | 1 | 10 | 10² | 10³ | 10⁴ | 10⁶ |
|---|------|------|------|---|---|----|----|----|----|
| share ÷ L2 | 1.000 | **1.000** | 0.999 | 0.987 | 0.883 | 0.452 | 0.134 | 0.081 | 0.075 |

(`python scripts/06_sobolev_argument.py` — no trained model, no torch, ~1 min. The
objective is within 1.3% of L2 on the cone up to λ = 1, i.e. 100× the default. The trained
objective's floor is 0.075 (~13×) at λ = 10⁶. The pure multiplicative reading, `(1+k²)^order`
as in neuraloperator's `H1Loss`, gives 0.053 (~19×); that is a different objective, not
reachable by the trained one, and what an earlier version of this file wrongly reported as
the trained one. A frequency-based fix cannot repair an amplitude-based blindness.)


## Reproduce

```bash
pip install -r requirements.txt
python scripts/00_validate_acoustics.py      # gate: known analytic answer
python scripts/04_komega_analysis.py         # the headline number — no model needed
python scripts/06_sobolev_argument.py        # why Sobolev cannot help — no model needed
python scripts/07_tform_metric.py            # T-form cone share R_T (includes sub-dk content)
python scripts/08_mach_scaling.py           # radiated power vs Mach: the compact 2D law
```

`scripts/01_generate_data.py` regenerates the dataset if `data/` is missing. The
headline figure and the Stage-4 argument need neither torch nor a GPU.

Training (optional, Stage 3):
```bash
python scripts/02_train_surrogate.py --model fno --out runs/fno_mse_01      # --smoke for a quick check
python scripts/03_compute_sources.py --model runs/fno_mse_01/model.pt --out runs/fno_mse_01   # field vs source vs far-field error
```

## The validation gate

`scripts/00` radiates a co-rotating vortex pair and checks two facts with known
answers: the tone is at **twice** the orbital frequency, and the far-field phase
winds with **azimuthal mode number 2**. Both pass to three decimals. CHECK 5 compares the
exact-kernel S-form and T-form far fields at the tone in L = 2π and 4π boxes: the gap is the
box surface term, 0.0526 → 0.0073, and the check asserts < 2% at 4π and a ≥ 3× fall with L.
Nothing downstream is trustworthy without this.

## Notes for anyone reading the code

Seven things in here were bugs and are worth knowing about if you write your own
version:

- **Parseval normalisation.** FFT power needs dividing by `N_total²`, not
  `N_total`. Invisible on a log-scale plot; inflates every energy by ~10⁵.
- **Radial bin centring.** Bins must be centred *on* multiples of `dk`, not
  starting at them, or a half-bin bias tilts the apparent phase speed and
  corrupts the radiating fraction.
- **Masking a binned spectrum.** Fixing the bin centring is not enough. A cone
  condition evaluated at bin *centres* tests every mode in a bin against the wrong
  `k`: on a square grid one bin holds several distinct `|k|` (5, √26 and √29 all
  land in the bin centred on 5). Near the cone edge that admits a slab of
  non-radiating modes — measured, it inflated the headline number by 2.7× (2.8× on
  a single full-record Hann estimate). The
  metric therefore runs on a radially *resolved* grid (`KOmegaSpectrum.k_resolved`),
  not the binned one. `radiating_mask` is kept for plotting only.
- **Rotating-quadrupole directivity.** A co-rotating pair radiates a *spinning*
  pattern, so `|p'|` at the tone frequency is nearly axisymmetric. Testing the
  magnitude for four lobes returns a near-null for a perfectly correct answer;
  the structure is in the phase.
- **Selecting a frequency band by array index.** `[i for i in range(...)][:24]`
  picks the 24 *lowest* frequencies, which at M = 0.1 are all below `c₀·dk` — the
  lowest frequency that can radiate at all. On `data/turb2d.npz`, **0 of 24** were
  supersonic, and 91% of the source's ω > 0 power is down there on the convective
  ridge. Select by the radiation condition; see `greens.select_band`.
- **The S-form far field in a periodic box.** The space-derivative form
  (Sarkar & Hussaini 1993, Eq. 1) includes a surface dipole from the nonzero velocity at the
  box edge, an artifact of the boundary conditions; the time-derivative form (Eq. 2)
  gives the quadrupole sound of a finite volume of turbulence in a medium at rest.
  Use the T-form (`greens.far_field_pressure_ibp`). On a vortex pair with net circulation the
  gap is 5.26% at L = 2π and falls with box size; on turbulence it was 7–168×.
- **Hard box truncation leaks into small k.** Cutting a finite box out of the stress puts
  convective-ridge content into low wavenumbers through the box sinc. Taper the box edge
  (Tukey 0.5) before estimating sub-`dk` radiation; see `tform.cone_share_refined`.

A sixth is a feature: the two Lighthill formulations (gradient form and double
divergence) are algebraically identical for solenoidal fields, so their
disagreement is a free **under-resolution detector** — 10⁻⁸ at n=96 with a
0.18 core, 10⁻¹⁴ at n=128.

## The open question

What objective *would* work? Not one built on `|k|` -- `scripts/06` shows the cone
is starved by where a `k`-based weight's mass is, not by its exponent, so the
weight has to be built from proximity to the sonic line in `(k, w)`. That needs a
time window to define `w`, so it is a multi-step objective, not a one-step one.
Deliberately left unimplemented in `src/models/losses.py` -- the diagnosis is worth
more intact than the fix.

## Layout

```
src/spectral.py            exact periodic derivatives, Helmholtz projection
src/acoustics/lighthill.py source term, two independent formulations
src/acoustics/komega.py    (k,w) spectra, sonic line, THE metric
src/acoustics/radiating.py loss weighting: L2 vs the trained Sobolev objective
src/acoustics/greens.py    2D Hankel Green's function, far field, radiating band
src/acoustics/tform.py     T-form cone share R_T (sub-dk included), radiated power
src/solvers/               2D NS (pseudospectral), vortex pair, IO
src/models/                FNO, UNet, losses, vanilla training
src/diagnostics/           spectra, spurious dilatation, the error triple
tests/                     64 tests, all passing
```
