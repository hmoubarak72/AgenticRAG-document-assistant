# Technical Note: Elastic versus Acoustic Full-Waveform Inversion

*Sample geophysics document for AgenticRAG-document-assistant. It summarizes widely used
industry and research practice; it does not report results from any specific survey.*

## Section 1 — Purpose

Full-waveform inversion updates a subsurface model by minimizing the misfit between recorded
and modeled seismic data. This note compares the two main formulations used in exploration
geophysics and gives practical guidance on choosing between them.

## Section 2 — How elastic FWI differs from acoustic FWI

Acoustic FWI solves the acoustic wave equation. It models pressure waves only and treats the
earth as a fluid, so the inverted parameters are P-wave velocity (Vp) and sometimes density.

Elastic FWI solves the elastic wave equation, which supports both compressional (P) and shear
(S) waves. It inverts for Vp, S-wave velocity (Vs) and density, or an equivalent set such as
the Lamé parameters or P- and S-impedance. Because shear stiffness is included, elastic
modeling reproduces effects that acoustic modeling cannot:

- P-to-S mode conversions at interfaces with a shear-velocity contrast
- S-waves and their reflections
- Surface waves (Rayleigh waves on land, Scholte waves at the seabed)
- The correct angle-dependent reflection amplitudes, that is, AVO effects

In acoustic FWI these elastic effects are not modeled and behave like coherent noise in the
data residual, which can bias the Vp model.

## Section 3 — Why elastic FWI costs more

Elastic modeling is more expensive for three reasons. First, the wavefield has more components:
a velocity-stress formulation propagates particle velocities and stresses instead of a single
pressure field, which increases compute and memory per grid point. Second, S-waves are slower
than P-waves, and the grid spacing must resolve the shortest wavelength, which is set by the
lowest S-wave velocity. In 3D finite-difference modeling the cost grows roughly with the fourth
power of maximum frequency divided by minimum velocity, so low near-surface or near-seafloor Vs
raises the cost sharply. Third, the inversion is harder: more parameters, stronger
inter-parameter crosstalk and more nonlinearity require more iterations and careful strategy.

## Section 4 — When the extra cost is justified

Elastic FWI is usually justified when elastic effects are strong in the data or when shear
information is the objective:

- Hard seabeds and strong contrasts such as salt, basalt or carbonates, where mode conversions
  and AVO effects are significant even in pressure (streamer) data.
- Soft seabeds recorded with ocean-bottom nodes or cables, where Scholte waves and P-to-S
  converted energy dominate the horizontal components.
- Land data, where surface waves carry most of the energy and constrain near-surface Vs, and
  where topography and the free surface must be modeled.
- Multicomponent data, where the horizontal geophone components record shear energy that only
  an elastic model can explain.
- Reservoir characterization, where Vs, the Vp/Vs ratio or Poisson's ratio help discriminate
  lithology and fluids.

## Section 5 — When acoustic FWI is usually sufficient

Acoustic FWI remains the standard choice for marine streamer data over soft or moderate seabeds
when the goal is a Vp model for imaging. Practical mitigations reduce elastic bias: invert
diving waves and early arrivals, limit offsets and angles where AVO effects grow, and down-weight
events that are clearly converted waves.

## Section 6 — Ocean-bottom node and multicomponent data

Ocean-bottom node (OBN) and ocean-bottom cable (OBC) surveys record pressure and three particle
velocity components. The vertical and horizontal components contain P-to-S converted waves and
Scholte waves that only elastic FWI can use. Vector fidelity and coupling of the horizontal
sensors must be checked, and the fluid-solid boundary at the seabed requires a coupled
acoustic-elastic solver.

## Section 7 — Practical workflow

A common strategy is hierarchical: build a reliable Vp model with acoustic FWI or tomography,
estimate a starting Vs from converted waves or surface-wave dispersion, then run elastic FWI
from low to high frequency. Density is inverted last, if at all, because it is weakly
constrained and couples strongly with Vp.

## Section 8 — Numerical methods for topography and the free surface

Land and shallow-water elastic FWI must model the free surface accurately because surface waves
are generated there. Spectral-element methods and curvilinear finite-difference grids follow
topography; a staircase approximation of topography on a regular grid generates artificial
scattering unless the grid is very fine. Absorbing boundaries such as convolutional perfectly
matched layers (C-PML) suppress reflections from the model edges.

## Section 9 — Decision checklist

1. Is the seabed hard, or are there strong elastic contrasts in the target interval?
2. Are there multicomponent or ocean-bottom data with converted waves?
3. Is the survey on land, with surface waves and rough topography?
4. Is Vs or the Vp/Vs ratio needed for reservoir characterization?
5. Is there a reliable Vp starting model and enough compute for the lowest Vs in the model?

If the answer to any of questions 1 to 4 is yes and question 5 is satisfied, elastic FWI is
likely worth its cost; otherwise acoustic FWI with elastic-aware data selection is usually
the more efficient choice.
