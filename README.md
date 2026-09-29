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

| Mach | L2 objective | H¹ (Sobolev) objective |
|------|--------------|------------------------|
| 0.05 | 8.9 × 10⁻¹⁰  | 7.0 × 10⁻¹¹ |
| 0.10 | 3.4 × 10⁻⁸   | 2.7 × 10⁻⁹  |
| 0.20 | 5.0 × 10⁻⁶   | 3.7 × 10⁻⁷  |
| 0.30 | 1.0 × 10⁻⁴   | 8.0 × 10⁻⁶  |

*(64² smoke-test run; regenerate at production resolution with `scripts/01` + `scripts/04`.)*

**Sobolev training — the standard remedy for spectral bias — makes it about ten
times worse.** It up-weights high `k`; the radiating cone is at low `k`. A
frequency-based fix cannot repair an amplitude-based blindness.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/00_validate_acoustics.py      # gate: known analytic answer
python scripts/01_generate_data.py --n 128 --snapshots 1024
python scripts/04_komega_analysis.py         # the headline number — no model needed
```

Training (optional, Stage 3):
```bash
python scripts/02_train_surrogate.py --model fno
python scripts/03_compute_sources.py         # field vs source vs far-field error
```

## The validation gate

`scripts/00` radiates a co-rotating vortex pair and checks two facts with known
answers: the tone is at **twice** the orbital frequency, and the far-field phase
winds with **azimuthal mode number 2**. Both pass to three decimals. Nothing
downstream is trustworthy without this.

## Notes for anyone reading the code

Three things in here were bugs in the first draft and are worth knowing about if
you write your own version:

- **Parseval normalisation.** FFT power needs dividing by `N_total²`, not
  `N_total`. Invisible on a log-scale plot; inflates every energy by ~10⁵.
- **Radial bin centring.** Bins must be centred *on* multiples of `dk`, not
  starting at them, or a half-bin bias tilts the apparent phase speed and
  corrupts the radiating fraction.
- **Rotating-quadrupole directivity.** A co-rotating pair radiates a *spinning*
  pattern, so `|p'|` at the tone frequency is nearly axisymmetric. Testing the
  magnitude for four lobes returns a near-null for a perfectly correct answer;
  the structure is in the phase.

A fourth is a feature: the two Lighthill formulations (gradient form and double
divergence) are algebraically identical for solenoidal fields, so their
disagreement is a free **under-resolution detector** — 10⁻⁸ at n=96 with a
0.18 core, 10⁻¹⁴ at n=128.

## Open question

What objective *would* work? The implied answer is weighting by proximity to the
sonic line in (k, ω) rather than by |k| — which requires a time window to define
ω, so it is a multi-step objective, not a one-step one. Deliberately left
unimplemented in `src/models/losses.py`.

## Layout

```
src/spectral.py            exact periodic derivatives, Helmholtz projection
src/acoustics/lighthill.py source term, two independent formulations
src/acoustics/komega.py    (k,ω) spectra, sonic line, THE metric
src/acoustics/greens.py    2D Hankel Green's function, far field
src/acoustics/radiating.py loss weighting: L2 vs Sobolev in the cone
src/solvers/               2D NS (pseudospectral), vortex pair, IO
src/models/                FNO, UNet, losses, vanilla training
src/diagnostics/           spectra, spurious dilatation, the error triple
tests/                     14 tests, all passing
```
