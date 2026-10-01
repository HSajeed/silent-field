# %% [markdown]
# # 00 — Overview and setup
#
# **silent-field** asks why neural surrogates of turbulent flow, trained with the usual
# L2 loss, get the *sound* of the flow wrong, and argues that the usual explanation
# (spectral bias) points the wrong way.
#
# This series of notebooks walks through the whole project in the order it happened:
#
# | # | chapter | what it covers | runtime (CPU) |
# |---|---|---|---|
# | 00 | **Overview and setup** (this one) | the claim, the physics, the repo, the timeline, tests | ~3 min |
# | 01 | Validation gate | a vortex pair with a known answer; the Green's-function fix | ~3 min |
# | 02 | Turbulence dataset | the solver, the data, stationarity, the −3.98 slope | ~5 min |
# | 03 | k–ω headline | the radiating cone and the number R | ~10 min |
# | 04 | Tier-0 fixes, old vs new | three bugs, run in both versions of the code | ~10 min |
# | 05 | Review findings (2026-10-01) | the far-field artifact, Stage-4 reframing, audit | ~10 min |
# | 06 | Stage 3 (future, partly runnable) | training a surrogate; the error triple | GPU ~1 h |
# | 07 | Stage 4 and roadmap (future, partly runnable) | Sobolev, causal test, plan | ~3 min |
#
# **How to run in Colab**
# 1. *Secrets* (key icon in the left bar): add `GH_TOKEN`, a GitHub token with read access
#    to `HSajeed/silent-field`, and enable notebook access.
# 2. *Google Drive*: the datasets go in `MyDrive/silent-field/data/`
#    (`turb2d.npz` is required; `turb2d_kf16.npz` and `turb2d_L4pi.npz` are optional).
# 3. Run the setup cell below. It clones the repo, mounts Drive and checks the data.
#
# Every chapter starts with the same setup cell and runs on its own. Chapters 03–05 share
# cached spectra (saved to `MyDrive/silent-field/cache/`), so running them in order is faster.

# %% [bootstrap]

# %% [markdown]
# ## 1. The claim
#
# *Spectral bias* says neural networks learn low wavenumbers first and lose high ones.
# The acoustic field of a low-Mach flow is at **low** wavenumber: the acoustic wavelength
# exceeds the flow's own length scale by a factor 1/M. By the spectral-bias argument it
# should be the easiest content to learn.
#
# The project's claim is that it is lost anyway, for a different reason: the part of the
# flow that radiates sound carries a vanishing share of the energy, and an L2 loss spends
# its effort where the energy is.
#
# > **Read this as the starting claim, not the conclusion.** Chapters 04 and 05 show how
# > the supporting numbers changed under scrutiny, and which parts are still open.

# %% [markdown]
# ## 2. The physics in five steps
#
# **(a) Which disturbances radiate.** A disturbance with wavevector **k** and angular
# frequency ω moves at phase speed ω/|k|. It reaches the acoustic far field only if that
# phase speed is supersonic:
#
# $$\frac{\omega}{|k|} \ge c_0 \quad\Longleftrightarrow\quad |k| \le \frac{\omega}{c_0}.$$
#
# In the (|k|, ω) plane this is the region above the **sonic line** ω = c₀|k| — the
# *radiating cone*.
#
# **(b) Where turbulence lives.** Eddies of size 1/|k| are swept along at roughly the flow
# speed U, so their energy sits near the **convective ridge** ω ≈ U|k|. Its slope is
# smaller than the sonic line's by exactly M = U/c₀. At M = 0.1 the two lines differ in
# slope by 10×.
#
# **(c) Where the sound comes from.** Lighthill's analogy rewrites the exact flow equations
# as a wave equation driven by the Reynolds stress:
#
# $$\Big(\frac{\partial^2}{\partial t^2} - c_0^2\nabla^2\Big)\rho' = \frac{\partial^2 T_{ij}}{\partial x_i\,\partial x_j},
# \qquad T_{ij}\approx \rho_0 u_i u_j \ \text{ at low Mach}.$$
#
# For an incompressible velocity field the source $S=\partial_i\partial_j(\rho_0 u_iu_j)$
# equals $\rho_0(\partial_j u_i)(\partial_i u_j)$, so it can be computed after the fact from
# an incompressible simulation. No compressible solver is needed. That is what makes the
# project cheap.
#
# **(d) How the sound reaches an observer, in 2D.** At each frequency the pressure is the
# source convolved with the 2D free-space Green's function. With numpy's $e^{+i\omega t}$
# convention the outgoing one is $G=(-i/4)H_0^{(2)}(kr)$, $k=\omega/c_0$. Its magnitude
# decays as $r^{-1/2}$ (not the 3D $r^{-1}$). Chapter 01 checks both facts.
#
# **(e) What an L2 loss sees.** By Parseval, an L2 loss is a sum over Fourier modes of
# squared amplitude. If the model's error were distributed like the energy, the share of
# the loss inside the cone would equal the share of the **energy** inside the cone. That
# share, **R**, is the project's headline number (chapter 03). The "if" matters: chapter 05
# explains why the loss share and the energy share can differ by orders of magnitude.

# %%
# The (|k|, omega) picture at M = 0.1, drawn to scale with the baseline dataset's U.
import numpy as np
import matplotlib.pyplot as plt

U, M = 1.1308, 0.1           # u_rms of data/turb2d.npz; Mach number
c0 = U / M
k = np.linspace(0, 64, 400)
omega_max = np.pi / 0.04     # Nyquist of the saved record (dt_save = 0.04)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.fill_between(k, c0 * k, omega_max, where=c0 * k <= omega_max, color=style.NEW, alpha=0.15,
                label=r"radiating cone  $\omega \geq c_0|k|$")
ax.plot(k, c0 * k, color=style.NEW, lw=2, label=r"sonic line  $\omega=c_0|k|$")
ax.plot(k, U * k, color=style.OLD, lw=2, ls="--", label=r"convective ridge  $\omega=U|k|$")
ax.axvline(1.0, color=style.REF, lw=0.8, ls=":")
ax.text(1.3, 70, "first box mode\n|k| = 1", fontsize=8, color=style.REF)
ax.set_xlim(0, 64); ax.set_ylim(0, omega_max)
ax.set_xlabel("|k|"); ax.set_ylabel(r"$\omega$")
ax.set_title("At M = 0.1 the cone is a thin sliver near |k| = 0")
ax.legend(loc="lower right")
plt.show()
print(f"c0 = {c0:.2f};  the cone reaches |k| = omega_max / c0 = {omega_max / c0:.2f}")

# %% [markdown]
# ## 3. The repository
#
# | path | role |
# |---|---|
# | `src/spectral.py` | the periodic grid and exact spectral derivatives |
# | `src/solvers/turb2d.py` | forced 2D Navier–Stokes, pseudospectral (chapter 02) |
# | `src/solvers/vortex_pair.py` | the co-rotating vortex pair used by the gate (chapter 01) |
# | `src/acoustics/lighthill.py` | the Lighthill source, two equivalent forms |
# | `src/acoustics/greens.py` | 2D Green's function, far-field pressure, frequency-band selection |
# | `src/acoustics/komega.py` | (k, ω) spectra and the radiating-cone mask |
# | `src/acoustics/radiating.py` | the headline metric and the Sobolev analysis |
# | `src/models/` | FNO, UNet, losses, a deliberately vanilla training loop |
# | `src/diagnostics/` | energy spectra, spurious divergence, the error triple |
# | `scripts/00`–`06` | gate, data generation, training, error triple, k–ω analysis, figures, Sobolev sweep |
# | `tests/` | unit tests |
# | `figures/` | stored outputs, including `komega_metrics.json` (the numbers) |
# | `runs/` | two archived analyses from 2026-09-15 (pre-Tier-0) |
# | `rerun_log_2026-09-15.md`, `TIER0_PLAN.md`, `branch_analysis.md`, `review_2026-10-01.md` | the audit trail |

# %% [markdown]
# ## 4. Timeline
#
# The chapters follow this sequence. "In git" says whether the code of that moment can
# be run again (chapter 04 runs the initial commit next to today's code). Steps before the
# first commit are shown in chapters 01 and 04 as clearly labelled *reconstructions*.

# %%
TIMELINE = [
    ("~09-14", "Build plan; Stages 1–2 written; README numbers from a 64² smoke run", "no", "00"),
    ("09-15", "Production rerun: gate PASS, baseline dataset, R = 8.7e-08 at M = 0.1", "no", "01–03"),
    ("09-15", "Independent audit: incoming Green's function; gate blind spots; cone energy at |k| = 1", "no", "01, 03"),
    ("09-15", "Follow-up B1–B8: kernel fixed; CHECKs 3–4; loss k-grid fixed; bands; kf16 and L4π datasets", "no", "01–04"),
    ("09-29", "Initial commit 2535bb4 (21 tests); branch_analysis.md review", "yes", "04"),
    ("09-30", "Tier 0: exact cone mask; Sobolev convention; far-field band; 42 tests", "yes", "04"),
    ("10-01", "Review: S-form far-field artifact; Stage-4 reframing; audit of branch_analysis", "yes", "05"),
    ("next", "Stage 3 (train surrogates, B metric), Stage 4, causal test", "-", "06, 07"),
]
md("| date (2026) | event | in git? | chapter |\n|---|---|---|---|\n" +
   "".join(f"| {d} | {e} | {g} | {c} |\n" for d, e, g, c in TIMELINE))

# %%
# Which commits exist in this clone.
print(subprocess.run(["git", "log", "--oneline", "--decorate", "-n", "10"],
                     capture_output=True, text=True).stdout)

# %% [markdown]
# ## 5. Check the installation
#
# The unit tests cover the solver, the Lighthill source, the Green's function, the
# (k, ω) spectrum, the cone mask, the losses, the Sobolev analysis and the notebook helpers.

# %%
out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--color=no"],
                     capture_output=True, text=True)
print(out.stdout[-1500:])
assert out.returncode == 0, "some tests failed -- see the output above"

# %% [markdown]
# ## 6. Check the data
#
# The datasets are not regenerated (the project rule is to reuse the files every logged
# number came from). Each file is checked two ways: its sha256 against the author's copy,
# and its stored metadata against the manifest tracked in git.

# %%
for name in nbenv.DATASETS:
    if not ENV.has(name):
        print(f"{name:<12} not present (optional)" if name != "turb2d" else
              f"{name:<12} MISSING -- required by chapters 02-07")
        continue
    info = nbenv.verify_dataset(ENV, name)
    print(f"{name:<12} OK  manifest {info['manifest']}  shape {info['shape']}  "
          f"L = {info['length']:.4f}  dt_save = {info['dt_save']}  u_rms = {info['u_rms']:.4f}")

# %% [markdown]
# ## What this chapter established
# - The claim, and the five pieces of physics it rests on.
# - Where each part lives in the code.
# - The order of events the next chapters follow.
# - The tests pass and the data is the data the logs describe.
