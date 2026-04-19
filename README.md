# NURBS-BEM EM Solver

A mesh-free, physics-based electromagnetic solver for solenoid 
geometries with ferromagnetic cores. Built from scratch to generate 
multi-fidelity training data for a surrogate model, bridging 
classical numerical methods and scientific machine learning.

> **Current status:** Solver complete · Dataset generated (25 
> solenoids, two fidelity levels) · Surrogate model in development

---

## Why This Exists

Engineering simulations of electromagnetic systems are expensive.
A single high-fidelity FEM run in ANSYS or COMSOL can take hours.
If you want to optimize a motor design, quantify uncertainty, or
explore a parameter space, you need to run that simulation
hundreds or thousands of times. That's computationally prohibitive.

The standard answer is a surrogate model: train a cheap 
mathematical approximation on a handful of simulations, then use
the surrogate for everything else. The problem is that surrogate
training requires data, and data requires simulations.

This solver is the data generation engine. It computes magnetic 
flux density B across a 3D field grid for arbitrary solenoid 
geometries, without generating a mesh in seconds instead of 
hours. The output dataset feeds a surrogate model that will 
eventually predict B-fields for unseen geometries at inference 
time.

The mesh-free design is deliberate. NURBS geometry representation
allows exact calculus on the geometry without discretization,
keeping the pipeline compatible with a planned NURBS-native
operator learning architecture.

---

## Physics

The total magnetic flux density is decomposed as:

B_total(r) = B_coil(r) + B_core(r)

### Coil Field — Biot-Savart Integration

The coil wire is modeled as a thin conductor carrying current I
along a NURBS centreline C(t). Analytic NURBS tangents C'(t) are
used — no finite differences:
B_coil(r) = (μ₀I / 4π) ∫ [C'(t)dt × R̂] / |R|²
Discretised over n_seg = clamp(50·N_turns, 200, 3000) segments.

### Core Field — Surface Bound Current BEM

For a linear isotropic magnetic core, volume bound currents vanish
and only surface bound currents K_s = M × n̂ contribute. The
material law gives M = χ·B_total where χ = (μᵣ−1)/(μ₀·μᵣ).

Substituting at each surface sample point i yields the
self-consistency equation:

Mᵢ = χ·B_coil(rᵢ) + χ·(μ₀/4π) Σⱼ Tᵢⱼ·Mⱼ

where Tᵢⱼ is the 3×3 influence block — the field at point i
per unit magnetisation at surface point j. Rearranging for all
N_S surface points simultaneously:

(I − χT) · m = χ · b_coil

This is a (3N_S) × (3N_S) linear system — default size 3000×3000
— solved by direct LU decomposition. Iterative solvers (GMRES,
Richardson) diverge for μᵣ > ~5 because the spectral radius of
χT exceeds 1. Direct solve achieves residual ||Am−b||/||b|| ≈ 10⁻¹⁵.

After solving for M, the core field at each grid point r_k is:

B_core(r_k) = (μ₀/4π) Σⱼ (K_sⱼ · dAⱼ) × R̂ⱼₖ / |Rⱼₖ|²

Full derivation: see `docs/em_solver_algorithm.pdf`

---

## Geometry Pipeline

All geometry is extracted from STEP files via OpenCASCADE (OCC).
No manual meshing required.


[1/6] Load STEP → identify coil + core solids by geometric scoring
[2/6] Coil: walk BRep edges → classify helix edges → centreline C_raw(t)
[3/6] NURBS coil fit: degree-3 NURBS C(t), n_ctrl = clamp(4·N_turns,12,120)
[4/6] Core: classify faces (lateral / end cap) → allocate N_S=1000 samples
[5/6] Core NURBS: lateral surface fit (degree 2,2) + OCC D1 end cap sampling
[6/6] Field solve: Grid → Biot-Savart → build T → solve → B_core → B_total

**Why NURBS?**
NURBS (Non-Uniform Rational B-Splines) represent geometry as 
piecewise polynomials defined by knot vectors and control points.
This allows exact calculus — integration and differentiation — 
directly on the geometry without discretization into a mesh. The
Biot-Savart integral in Step 6 uses analytic NURBS derivatives,
not numerical finite differences. This is the key advantage over
mesh-based approaches: the geometry stays in its natural continuous
representation throughout the pipeline.

---

## Multi-Fidelity Dataset

The solver produces data at two fidelity levels. This structure
is intentional — it enables multifidelity surrogate training
where low-fidelity data augments scarce high-fidelity runs.

| Level | Grid N | Solve Time | Accuracy vs FEM | Role |
|-------|--------|------------|-----------------|------|
| Low   | N=20   | ~10s       | ~60–80%         | Cheap, abundant |
| High  | N=50   | ~60s       | ~85–90%         | Expensive, scarce |
| FEM   | —      | hours      | ground truth    | Future benchmark |

**Dataset summary (25 solenoid geometries):**

| Parameter              | Range          |
|------------------------|----------------|
| Wire current I         | 0.5 – 5.0 A    |
| Frequency f            | 50 – 1000 Hz   |
| Relative permeability  | 35 – 3000      |
| Grid resolution N      | 20 (low), 50 (high) |

**Outputs per solenoid:**
- `field_total.npz` — Bx, By, Bz, |B|, X, Y, Z arrays on N³ grid
- `metadata.json` — B_max, B_mean, all physics parameters
- `coil_nurbs.json` — NURBS control points and knot vectors
- `core_nurbs.json` — Core surface NURBS representation
- `summary.png` — Field visualization

