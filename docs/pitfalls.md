# Pitfalls found while building this

Things that were bugs in this pipeline. Worth knowing if you write your own version.

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

An eighth entry is a feature: the two Lighthill formulations (gradient form and double
divergence) are algebraically identical for solenoidal fields, so their
disagreement is a free **under-resolution detector** — 1.2 × 10⁻⁸ at n=96 with a
0.18 core, 7 × 10⁻¹⁵ at n=128.
