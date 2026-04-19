"""
em_analysis.py  —  NURBS EM Field Solver
==========================================
Coil : NURBS curve C(t)     fitted to centreline (auto-parameterised)
Core : NURBS surface S(u,v)  fitted to OCC D1 samples (auto-parameterised)

All NURBS resolution parameters are derived automatically from the
imported STEP geometry. The user only provides physics inputs.

Run interactively:
    python em_analysis.py

Or with arguments:
    python em_analysis.py --step solenoid_01.step --current 1.0
                          --frequency 1000 --mu_r 150 --sigma_core 0.01

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EQUATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] NURBS COIL CURVE
    C(t) = Σ N_{i,3}(t) · P_i^coil
    C'(t) = Σ N'_{i,3}(t) · P_i^coil   (analytic derivative via geomdl)

[2] COIL BIOT-SAVART (thin-wire line integral)
    B_coil(r) = (μ₀I/4π) Σᵢ [ C'(tᵢ)·Δt × R̂ᵢ ] / |Rᵢ|²
    where Rᵢ = r − C(tᵢ),  R̂ᵢ = Rᵢ/|Rᵢ|

[3] NURBS CORE SURFACE
    S(u,v) = Σᵢ Σⱼ N_{i,2}(u) N_{j,2}(v) · P_{ij}^core
    ∂S/∂u, ∂S/∂v via central FD on NURBS parameters

[4] CORE SURFACE NORMAL (from fitted NURBS)
    n̂(u,v) = (∂S/∂u × ∂S/∂v) / |∂S/∂u × ∂S/∂v|

[5] MAGNETIZATION (linear homogeneous core → Jm = 0)
    M(r) = (μᵣ−1)/μ₀ · B(r)   inside core
    M(r) = 0                  outside core

[6] SURFACE BOUND CURRENT on S(u,v)
    Ks(u,v) = M(u,v) × n̂(u,v)

[7] CORE BIOT-SAVART (surface bound current)
    B_core(r) = (μ₀/4π) Σᵢⱼ [ Ks(uᵢ,vⱼ) × R̂ᵢⱼ ] / |Rᵢⱼ|² · dA

[8] SELF-CONSISTENT ITERATION
    B^(k+1) = B_coil + B_core[ M(B^(k)) ]
    converged when ‖B^(k+1)−B^(k)‖ / ‖B^(k+1)‖ < tol

[9] SKIN DEPTH
    δ_wire = √(1 / π·f·μ₀·σ_Cu)
    δ_core = √(1 / π·f·μ₀·μᵣ·σ_core)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dependencies
------------
    pip install numpy matplotlib geomdl
    conda install -c conda-forge pythonocc-core
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# OCC — used ONLY during geometry import and NURBS fitting
from OCC.Core.STEPControl  import STEPControl_Reader
from OCC.Core.TopExp       import TopExp_Explorer
from OCC.Core.TopAbs       import (TopAbs_SOLID, TopAbs_FACE, TopAbs_FORWARD,
                                   TopAbs_EDGE, TopAbs_WIRE, TopAbs_SHELL)
from OCC.Core.BRep         import BRep_Tool
from OCC.Core.BRepMesh     import BRepMesh_IncrementalMesh
from OCC.Core.BRepGProp    import brepgprop
from OCC.Core.GProp        import GProp_GProps
from OCC.Core.TopoDS       import topods
from OCC.Core.Bnd          import Bnd_Box
from OCC.Core.BRepBndLib   import brepbndlib
from OCC.Core.BRepAdaptor  import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.gp           import gp_Pnt, gp_Vec
from OCC.Core.GeomAbs      import (GeomAbs_Line, GeomAbs_Circle,
                                   GeomAbs_BSplineCurve, GeomAbs_BezierCurve,
                                   GeomAbs_OtherCurve)

# geomdl — used for ALL physics evaluation after fitting
from geomdl import BSpline, fitting, operations

MU0      = 4.0 * np.pi * 1e-7
SIGMA_CU = 5.8e7


# ═════════════════════════════════════════════════════════════════════════════
# 1.  USER INPUT PROMPTS  (physics only — geometry is auto-detected)
# ═════════════════════════════════════════════════════════════════════════════

def prompt_inputs(args):
    print("\n" + "="*56)
    print("  EM Analysis — NURBS Solver")
    print("  (NURBS parameters auto-detected from geometry)")
    print("="*56)

    # Geometry
    print("\n  -- Geometry --")
    if not args.step:
        while True:
            p = input("  Path to STEP file: ").strip()
            if os.path.isfile(p):
                args.step = p
                break
            print(f"    File not found: {p}")

    # Electrical
    print("\n  -- Electrical --")
    if args.current is None:
        while True:
            try:
                args.current = float(input("  Wire current I [A]: "))
                break
            except ValueError:
                print("    Please enter a number.")

    if args.frequency is None:
        while True:
            try:
                args.frequency = float(input("  Operating frequency f [Hz]: "))
                break
            except ValueError:
                print("    Please enter a number.")

    # Core magnetic properties
    print("\n  -- Core magnetic properties --")
    print("  (air core: mu_r=1, sigma_core=0)")
    if args.mu_r is None:
        while True:
            try:
                args.mu_r = float(input("  Relative permeability mu_r: "))
                break
            except ValueError:
                print("    Please enter a number.")

    if args.sigma_core is None:
        while True:
            try:
                args.sigma_core = float(
                    input("  Core conductivity sigma_core [S/m]\n"
                          "    (ferrite ~0.01, iron powder ~1000, air=0): "))
                break
            except ValueError:
                print("    Please enter a number.")

    # Field grid — only tunable parameter left
    print("\n  -- Field grid --")
    if args.grid_n is None:
        v = input("  Grid points per axis N [20]: ").strip()
        args.grid_n = int(v) if v else 20

    # Solver
    print("\n  -- Solver --")
    if args.tol is None:
        v = input("  Convergence tolerance [1e-3]: ").strip()
        args.tol = float(v) if v else 1e-3
    if args.max_iter is None:
        v = input("  Max iterations [30]: ").strip()
        args.max_iter = int(v) if v else 30

    return args


# ═════════════════════════════════════════════════════════════════════════════
# 2.  STEP LOADING  (OCC — import only)
# ═════════════════════════════════════════════════════════════════════════════

def load_step(path):
    r = STEPControl_Reader()
    if r.ReadFile(path) != 1:
        raise IOError(f"Cannot read STEP: {path}")
    r.TransferRoots()
    return r.OneShape()

def get_solids(shape):
    solids, ex = [], TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        solids.append(topods.Solid(ex.Current()))
        ex.Next()
    return solids

def solid_volume(solid):
    p = GProp_GProps()
    brepgprop.VolumeProperties(solid, p)
    return abs(p.Mass())

def bbox(shape):
    b = Bnd_Box()
    brepbndlib.Add(shape, b)
    v = b.Get()
    return np.array(v[:3]), np.array(v[3:])

def solid_surface_area(solid):
    p = GProp_GProps()
    brepgprop.SurfaceProperties(solid, p)
    return abs(p.Mass())

def identify_solids(solids):
    """
    Identify coil and core using multiple shape descriptors.

    A helical wire and a cylinder have very different geometric signatures:

    Metric                  Coil wire       Core cylinder
    ──────────────────────  ─────────────   ─────────────
    SA / Volume ratio       HIGH            LOW
    (surface area per unit volume — wire wraps around, huge surface)

    Bounding box compactness LOW             HIGH
    (how close to a cube — helix is elongated and spread out)
    compactness = min(dims) / max(dims)

    XY spread / bbox XY     HIGH            LOW
    (point cloud fills the XY annulus for a helix,
     clusters near centre for a cylinder)

    Each metric votes for which solid is the coil.
    The solid with the most votes is identified as the coil.
    """
    if len(solids) == 1:
        print("  Only 1 solid — treating as coil, no core")
        return solids[0], None

    scores = np.zeros(len(solids))   # higher score = more likely to be coil

    vols   = []
    areas  = []
    ratios = []
    compacts = []
    xy_spreads = []

    for i, solid in enumerate(solids):
        vol  = solid_volume(solid)
        area = solid_surface_area(solid)
        vols.append(vol)
        areas.append(area)

        # Metric 1: SA/V ratio — coil is HIGH
        ratio = area / max(vol, 1e-12)
        ratios.append(ratio)

        # Metric 2: bounding box compactness — coil is LOW
        pmin, pmax = bbox(solid)
        dims = pmax - pmin
        compact = float(np.min(dims)) / float(max(np.max(dims), 1e-12))
        compacts.append(compact)

        # Metric 3: XY point spread relative to bbox XY size
        # Tessellate lightly just for shape analysis
        try:
            pts     = tessellate(solid, deflection=0.5)
            xy_std  = float(np.std(pts[:,0]) + np.std(pts[:,1]))
            xy_bbox = float(max(dims[0], dims[1]))
            xy_spread = xy_std / max(xy_bbox, 1e-12)
        except Exception:
            xy_spread = 0.5
        xy_spreads.append(xy_spread)

    # Normalise each metric to [0,1] across solids and vote
    def norm(vals):
        v = np.array(vals, dtype=float)
        span = v.max() - v.min()
        if span < 1e-12:
            return np.zeros(len(v))
        return (v - v.min()) / span

    # Coil scores HIGH on SA/V, LOW on compactness, HIGH on XY spread
    scores += norm(ratios)          # higher ratio → coil
    scores += 1.0 - norm(compacts)  # lower compactness → coil
    scores += norm(xy_spreads)      # higher XY spread → coil

    coil_idx = int(np.argmax(scores))
    core_idx = int(np.argmin(scores))

    print(f"  Solid identification scores:")
    for i in range(len(solids)):
        tag = ""
        if i == coil_idx: tag = " [COIL]"
        if i == core_idx: tag = " [CORE]"
        print(f"    Solid {i}: vol={vols[i]:.3e}  "
              f"SA/V={ratios[i]:.2f}  "
              f"compact={compacts[i]:.3f}  "
              f"XY_spread={xy_spreads[i]:.3f}  "
              f"score={scores[i]:.3f}{tag}")

    coil = solids[coil_idx]
    core = solids[core_idx] if coil_idx != core_idx else None
    return coil, core

def tessellate(solid, deflection=0.1):
    """Triangle centroid point cloud from solid faces."""
    BRepMesh_IncrementalMesh(solid, deflection).Perform()
    pts, ex = [], TopExp_Explorer(solid, TopAbs_FACE)
    while ex.More():
        face = topods.Face(ex.Current())
        loc  = face.Location()
        tri  = BRep_Tool.Triangulation(face, loc)
        if tri:
            for i in range(1, tri.NbTriangles() + 1):
                n1, n2, n3 = tri.Triangle(i).Get()
                v1 = tri.Node(n1); v2 = tri.Node(n2); v3 = tri.Node(n3)
                pts.append([(v1.X()+v2.X()+v3.X())/3,
                             (v1.Y()+v2.Y()+v3.Y())/3,
                             (v1.Z()+v2.Z()+v3.Z())/3])
        ex.Next()
    if not pts:
        raise RuntimeError("Tessellation returned no points.")
    return np.array(pts)


def extract_centreline_from_edges(coil_solid, n_samples=500):
    """
    Extract the true helix centreline directly from the B-spline edge
    curves stored in the coil solid BRep topology.

    Why this works
    ──────────────
    The solenoid wire is a swept solid. Its BRep contains:
      - Helical B-spline edges tracing the outer and inner edges of the wire
      - Short connecting edges at the wire ends (caps, entry/exit)
      - Flat/circular end cap edges

    The helical edges span the full coil height and have large XY extent.
    The centreline = mean of all helix edge sample points at the same
    arc-length parameter.

    Strategy
    ────────
    1. Walk all edges in the coil solid via TopExp_Explorer
    2. For each edge, evaluate BRepAdaptor_Curve at uniform parameter steps
    3. Classify as helix edge if:
         - Z span > 50% of coil height
         - XY span > 30% of coil height
         - Radius R = sqrt(X²+Y²) is roughly uniform (R_std/R_mean < 0.15)
    4. Sample all helix edges at the same n_samples parameter values
    5. Average their XYZ positions → centreline

    This gives the true geometric centreline of the wire regardless of
    the tessellation density.
    """
    pmin, pmax  = bbox(coil_solid)
    coil_height = pmax[2] - pmin[2]
    coil_axis   = int(np.argmax(pmax - pmin))

    # Walk all edges
    helix_edges = []
    ex = TopExp_Explorer(coil_solid, TopAbs_EDGE)
    seen = set()
    while ex.More():
        edge = topods.Edge(ex.Current())
        # Deduplicate by hash
        h = edge.__hash__()
        if h in seen:
            ex.Next()
            continue
        seen.add(h)

        try:
            adaptor = BRepAdaptor_Curve(edge)
            t1      = adaptor.FirstParameter()
            t2      = adaptor.LastParameter()
            if abs(t2 - t1) < 1e-10:
                ex.Next()
                continue

            # Sample edge at 50 pts for classification
            ts      = np.linspace(t1, t2, 50)
            pts_e   = np.array([[adaptor.Value(t).X(),
                                  adaptor.Value(t).Y(),
                                  adaptor.Value(t).Z()] for t in ts])

            dims_e  = pts_e.max(axis=0) - pts_e.min(axis=0)
            z_span  = float(dims_e[2])
            xy_span = float(max(dims_e[0], dims_e[1]))
            R       = np.sqrt(pts_e[:,0]**2 + pts_e[:,1]**2)
            R_mean  = float(R.mean())
            R_std   = float(R.std())
            R_rel   = R_std / max(R_mean, 1e-6)

            # Helix edge: spans most of coil height, has XY extent,
            # radius is roughly constant
            if (z_span   > coil_height * 0.5 and
                xy_span  > coil_height * 0.3 and
                R_rel    < 0.20):
                helix_edges.append((adaptor, t1, t2, R_mean, z_span))
                print(f"    Helix edge: z_span={z_span:.2f}  "
                      f"xy_span={xy_span:.2f}  R={R_mean:.2f}±{R_std:.3f}")
        except Exception:
            pass
        ex.Next()

    if not helix_edges:
        raise RuntimeError(
            "No helix edges found in coil solid.\n"
            "  Cannot extract centreline from BRep edges.\n"
            "  Check that the correct solid was identified as the coil.")

    print(f"  Found {len(helix_edges)} helix edge(s)")

    # Sample all helix edges at n_samples uniform parameter values
    # and average their positions to get the centreline
    all_sampled = []
    for (adaptor, t1, t2, R_mean, z_span) in helix_edges:
        ts  = np.linspace(t1, t2, n_samples)
        pts = np.array([[adaptor.Value(t).X(),
                          adaptor.Value(t).Y(),
                          adaptor.Value(t).Z()] for t in ts])
        all_sampled.append(pts)

    # Average across all helix edges → true centreline
    # Sort each edge's sample points by Z before averaging so they align
    for i in range(len(all_sampled)):
        idx = np.argsort(all_sampled[i][:, 2])
        all_sampled[i] = all_sampled[i][idx]

    centreline = np.mean(all_sampled, axis=0)   # (n_samples, 3)

    # Sort by Z to ensure monotonic ordering
    idx = np.argsort(centreline[:, 2])
    centreline = centreline[idx]

    print(f"  Centreline: {len(centreline)} pts  "
          f"Z=[{centreline[:,2].min():.2f}, {centreline[:,2].max():.2f}]  "
          f"R_mean={np.sqrt(centreline[:,0]**2+centreline[:,1]**2).mean():.2f}")

    return centreline


# ═════════════════════════════════════════════════════════════════════════════
# 3.  GEOMETRY ANALYSIS  — auto-detect NURBS parameters from shape
# ═════════════════════════════════════════════════════════════════════════════

def analyse_coil_geometry(coil_solid):
    """
    Derive all coil NURBS parameters directly from BRep edge topology.
    Centreline extracted from helix edges via OCC BRepAdaptor_Curve.
    """
    pmin, pmax = bbox(coil_solid)
    dims       = pmax - pmin

    print(f"  Bounding box X : {pmin[0]:.3f} to {pmax[0]:.3f}  (span {dims[0]:.3f})")
    print(f"  Bounding box Y : {pmin[1]:.3f} to {pmax[1]:.3f}  (span {dims[1]:.3f})")
    print(f"  Bounding box Z : {pmin[2]:.3f} to {pmax[2]:.3f}  (span {dims[2]:.3f})")

    print(f"  Extracting centreline from BRep helix edges ...")
    cl = extract_centreline_from_edges(coil_solid, n_samples=500)

    R_cl   = np.sqrt(cl[:,0]**2 + cl[:,1]**2)
    coil_r = float(R_cl.mean())
    # Height is always Z span — coil axis is always Z
    # (using argmax was wrong for wide flat coils where X/Y span > Z span)
    height = float(dims[2])

    # Wire diameter: use bbox outer radius minus centreline radius.
    # This is accurate because the helix edges are on the outer surface of the wire.
    # Fallback: 1% of height (never less than that to avoid division issues).
    outer_r = float(max(dims[0], dims[1])) / 2.0
    wire_d_from_bbox = outer_r - coil_r
    # Also estimate from Z bbox: wire_d ~ pitch = height / n_turns (after n_turns known)
    # We use bbox method here; pitch method is a cross-check after n_turns is computed.
    wire_d  = max(wire_d_from_bbox, height * 0.01)

    theta     = np.arctan2(cl[:,1], cl[:,0])
    theta_uw  = np.unwrap(theta)
    total_rot = abs(theta_uw[-1] - theta_uw[0])
    n_turns   = max(int(round(total_rot / (2 * np.pi))), 1)

    n_ctrl = min(max(n_turns * 4,  12), 120)
    n_seg  = min(max(n_turns * 50, 200), 3000)

    # Cross-check wire_d against pitch (height / n_turns).
    # Best estimate: average of bbox method and pitch method,
    # since bbox slightly underestimates and pitch slightly overestimates.
    pitch  = height / max(n_turns, 1)
    wire_d = (wire_d + pitch) / 2.0

    print(f"  Coil radius    : {coil_r:.3f}")
    print(f"  Coil height    : {height:.3f}")
    print(f"  Wire diameter  : {wire_d:.3f}  (pitch={pitch:.3f})")
    print(f"  Est. turns     : {n_turns}  ({total_rot/(2*np.pi):.1f} revolutions)")
    print(f"  → n_ctrl       : {n_ctrl}")
    print(f"  → n_seg (BS)   : {n_seg}")

    return dict(
        cl      = cl,
        wire_d  = wire_d,
        coil_r  = coil_r,
        height  = height,
        n_turns = n_turns,
        n_ctrl  = n_ctrl,
        n_seg   = n_seg,
    )


def analyse_core_geometry(core_solid):
    """
    Analyse the core solid to derive NURBS surface parameters automatically.

    Detection strategy
    ──────────────────
    1. Bounding box → core radius, height
    2. Circumference = 2π * radius
    3. n_samp_u = max(32, int(circumference / 1.0))   ~1 sample per mm azimuth
       n_samp_v = max(16, int(height / 1.0))           ~1 sample per mm height
       Both capped at 64
    4. n_ctrl_u = max(8,  n_samp_u // 4)
       n_ctrl_v = max(6,  n_samp_v // 4)
       Both capped at 16

    Returns dict of geometry parameters
    """
    pmin, pmax = bbox(core_solid)
    dims       = pmax - pmin

    core_r  = float(max(dims[0], dims[1])) / 2.0
    height  = float(dims[2])
    circum  = 2.0 * np.pi * core_r

    n_su    = int(np.clip(circum / 1.0, 16, 64))
    n_sv    = int(np.clip(height  / 1.0, 8,  64))
    n_cu    = int(np.clip(n_su // 4,     4,  16))
    n_cv    = int(np.clip(n_sv // 4,     4,  16))

    print(f"  Core radius    : {core_r:.3f}")
    print(f"  Core height    : {height:.3f}")
    print(f"  → n_samp_u     : {n_su}")
    print(f"  → n_samp_v     : {n_sv}")
    print(f"  → n_ctrl_u     : {n_cu}")
    print(f"  → n_ctrl_v     : {n_cv}")

    return dict(
        core_r = core_r,
        height = height,
        n_su   = n_su,
        n_sv   = n_sv,
        n_cu   = n_cu,
        n_cv   = n_cv,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4.  NURBS COIL CURVE  C(t)
#     C(t) = Σ N_{i,3}(t) · P_i^coil
# ═════════════════════════════════════════════════════════════════════════════

def fit_coil_nurbs(cl, n_ctrl, degree=3):
    """
    Fit a degree-3 NURBS curve through the centreline points cl.
    cl is already extracted and ordered by analyse_coil_geometry.
    """
    if len(cl) < degree + 1:
        raise RuntimeError(
            f"Only {len(cl)} centreline points — cannot fit degree-{degree} NURBS.")

    n_ctrl = min(max(degree + 1, n_ctrl), len(cl))
    curve  = fitting.approximate_curve(
        cl.tolist(), degree=degree, ctrlpts_size=n_ctrl)

    print(f"  NURBS degree   : {curve.degree}")
    print(f"  Control pts    : {len(curve.ctrlpts)} × 3")
    return curve, cl


# ═════════════════════════════════════════════════════════════════════════════
# 5.  NURBS CORE SURFACE  S(u,v)
#     S(u,v) = Σᵢ Σⱼ N_{i,2}(u) N_{j,2}(v) · P_{ij}^core
# ═════════════════════════════════════════════════════════════════════════════

def sample_face_direct(adaptor, area, n_u, n_v, flip_normal=False):
    """
    Sample a single BRep face directly via OCC D1.
    Returns positions (N,3), normals (N,3), dA scalar for this face.
    Uses direct OCC sampling — no NURBS fitting needed for flat end caps.
    """
    umin = adaptor.FirstUParameter()
    umax = adaptor.LastUParameter()
    vmin = adaptor.FirstVParameter()
    vmax = adaptor.LastVParameter()

    positions, normals = [], []
    for u in np.linspace(umin, umax, n_u):
        for v in np.linspace(vmin, vmax, n_v):
            P  = gp_Pnt(); dU = gp_Vec(); dV = gp_Vec()
            try:
                adaptor.D1(u, v, P, dU, dV)
            except Exception:
                continue
            pt = np.array([P.X(), P.Y(), P.Z()])
            tu = np.array([dU.X(), dU.Y(), dU.Z()])
            tv = np.array([dV.X(), dV.Y(), dV.Z()])
            n  = np.cross(tu, tv)
            mag = np.linalg.norm(n)
            if mag < 1e-12:
                continue
            n = n / mag
            if flip_normal:
                n = -n
            positions.append(pt)
            normals.append(n)

    if not positions:
        return np.zeros((0,3)), np.zeros((0,3)), 0.0

    pts = np.array(positions)
    nrm = np.array(normals)
    dA  = area / len(pts)
    return pts, nrm, dA


def fit_core_nurbs(core_solid, n_su, n_sv, n_cu, n_cv, degree=2):
    """
    Sample ALL core BRep faces (lateral + top + bottom end caps).

    Each face is sampled directly via OCC D1. The lateral face uses
    NURBS fitting for smooth normals; end caps are sampled directly
    since they are flat (normal is constant ±Z).

    Returns: all_pts (N,3), all_normals (N,3), all_dA (N,) — per-point dA.
    total_area is returned for diagnostics only.

    Scalable: works for any core shape — tall, flat, or anything in between.
    """
    # Collect all faces with their areas
    faces_data = []
    total_area = 0.0
    ex = TopExp_Explorer(core_solid, TopAbs_FACE)
    while ex.More():
        face    = topods.Face(ex.Current())
        adaptor = BRepAdaptor_Surface(face, True)
        p       = GProp_GProps()
        brepgprop.SurfaceProperties(face, p)
        area    = abs(p.Mass())
        is_fwd  = (face.Orientation() == TopAbs_FORWARD)
        faces_data.append((adaptor, is_fwd, area))
        total_area += area
        ex.Next()

    if not faces_data:
        raise RuntimeError("No faces found on core solid.")

    # Sort by area — largest face first (lateral surface)
    faces_data.sort(key=lambda x: x[2], reverse=True)
    n_faces = len(faces_data)
    print(f"  Core faces     : {n_faces}  total area={total_area:.4e}")

    all_pts, all_nrm, all_dA = [], [], []

    NS_MAX = 1000   # total surface sample budget — keeps matrix at ~9M entries
                    # and solve time under ~60s for any geometry

    # Classify all faces as lateral (curved) or end cap (flat) by Z-extent
    classified = []
    for adaptor, is_fwd, area in faces_data:
        umin = adaptor.FirstUParameter(); umax = adaptor.LastUParameter()
        vmin = adaptor.FirstVParameter(); vmax = adaptor.LastVParameter()
        test_pts = []
        for u in np.linspace(umin, umax, 5):
            for v in np.linspace(vmin, vmax, 5):
                P = gp_Pnt(); dU = gp_Vec(); dV = gp_Vec()
                try:
                    adaptor.D1(u, v, P, dU, dV)
                    test_pts.append([P.X(), P.Y(), P.Z()])
                except Exception:
                    pass
        if not test_pts:
            continue
        tp = np.array(test_pts)
        z_range = tp[:,2].max() - tp[:,2].min()
        is_endcap = z_range < 1.0
        classified.append((adaptor, is_fwd, area, 'endcap' if is_endcap else 'lateral'))

    lateral_faces = [(a,f,ar) for a,f,ar,t in classified if t == 'lateral']
    endcap_faces  = [(a,f,ar) for a,f,ar,t in classified if t == 'endcap']

    if not lateral_faces:
        lateral_faces = [(faces_data[0][0], faces_data[0][1], faces_data[0][2])]
        endcap_faces  = [(a,f,ar) for a,f,ar,_ in classified[1:]]

    print(f"  Face types: {len(lateral_faces)} lateral, {len(endcap_faces)} end caps")

    # Allocate NS_MAX samples proportionally by face area
    all_face_areas = [ar for _,_,ar in lateral_faces] + [ar for _,_,ar in endcap_faces]
    total_surf_area = sum(all_face_areas)
    def pts_for_face(area):
        return max(64, int(NS_MAX * area / total_surf_area))

    # ── Lateral face: NURBS fit then evaluate ──────────────────────────────
    for li, (adaptor, is_fwd, area) in enumerate(lateral_faces):
        n_alloc = pts_for_face(area)
        n_side  = max(degree+2, int(np.sqrt(n_alloc)))
        umin = adaptor.FirstUParameter(); umax = adaptor.LastUParameter()
        vmin = adaptor.FirstVParameter(); vmax = adaptor.LastVParameter()
        grid_pts = []
        for u in np.linspace(umin, umax, n_side):
            for v in np.linspace(vmin, vmax, n_side):
                P = gp_Pnt(); dU = gp_Vec(); dV = gp_Vec()
                try:
                    adaptor.D1(u, v, P, dU, dV)
                    grid_pts.append([P.X(), P.Y(), P.Z()])
                except Exception:
                    pass
        n_cu_c = min(max(degree+1, n_cu), n_side)
        n_cv_c = min(max(degree+1, n_cv), n_side)
        surf = fitting.approximate_surface(
            grid_pts, size_u=n_side, size_v=n_side,
            degree_u=degree, degree_v=degree,
            ctrlpts_size_u=n_cu_c, ctrlpts_size_v=n_cv_c,
        )
        print(f"  Lateral {li} [area={area:.3e}, alloc={n_alloc}pts]: NURBS {n_side}x{n_side}")
        pts_f, nrm_f = eval_nurbs_surface(surf, nu=n_side, nv=n_side)
        dA_f = area / max(len(pts_f), 1)
        all_pts.append(pts_f); all_nrm.append(nrm_f)
        all_dA.append(np.full(len(pts_f), dA_f))

    # ── End caps: direct OCC sampling ─────────────────────────────────────
    for fi, (adaptor, is_fwd, area) in enumerate(endcap_faces):
        n_alloc = pts_for_face(area)
        n_side  = max(8, int(np.sqrt(n_alloc)))
        pts_f, nrm_f, dA_f = sample_face_direct(
            adaptor, area, n_side, n_side, flip_normal=not is_fwd)
        print(f"  End cap {fi} [area={area:.3e}, alloc={n_alloc}pts]: direct {n_side}x{n_side} = {len(pts_f)} pts")
        if len(pts_f) > 0:
            all_pts.append(pts_f); all_nrm.append(nrm_f)
            all_dA.append(np.full(len(pts_f), dA_f))

    all_pts = np.vstack(all_pts)
    all_nrm = np.vstack(all_nrm)
    all_dA  = np.concatenate(all_dA)

    print(f"  Total surface samples: {len(all_pts)}  (lateral + {n_faces-1} other faces)")
    return all_pts, all_nrm, all_dA, total_area


# ═════════════════════════════════════════════════════════════════════════════
# 6.  NURBS SURFACE EVALUATION  (geomdl only — no OCC after fitting)
#     Equations [3] and [4]
# ═════════════════════════════════════════════════════════════════════════════

def eval_nurbs_surface(surf, nu, nv):
    """
    Evaluate fitted NURBS surface S(u,v) at (nu×nv) parameter points.

    Position    : S(u,v)         geomdl evaluate_single
    Partials    : ∂S/∂u, ∂S/∂v  central FD on NURBS parameters
    Normal [4]  : n̂ = (∂S/∂u × ∂S/∂v) / |∂S/∂u × ∂S/∂v|

    No OCC used here — pure geomdl.
    """
    eps = 1e-5
    positions, normals = [], []

    for u in np.linspace(0., 1., nu):
        for v in np.linspace(0., 1., nv):
            pt   = np.array(surf.evaluate_single((u, v)))

            u1   = min(u + eps, 1.); u0 = max(u - eps, 0.)
            v1   = min(v + eps, 1.); v0 = max(v - eps, 0.)

            dSdu = (np.array(surf.evaluate_single((u1, v))) -
                    np.array(surf.evaluate_single((u0, v)))) / (u1 - u0)
            dSdv = (np.array(surf.evaluate_single((u, v1))) -
                    np.array(surf.evaluate_single((u, v0)))) / (v1 - v0)

            n    = np.cross(dSdu, dSdv)
            mag  = np.linalg.norm(n)
            if mag < 1e-12:
                continue
            n = n / mag

            positions.append(pt)
            normals.append(n)

    return np.array(positions), np.array(normals)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  COIL BIOT-SAVART  —  NURBS analytic tangent C'(t)
#     Equations [1] and [2]
# ═════════════════════════════════════════════════════════════════════════════

def biot_savart_coil(curve, current, field_pts, n_seg, wire_radius):
    """
    Thin-wire Biot-Savart along NURBS coil curve C(t).

    Equation [2]:
        B_coil(r) = (μ₀I/4π) Σᵢ [ C'(tᵢ)·Δt × R̂ᵢ ] / |Rᵢ|²

    C'(tᵢ) is the analytic NURBS first derivative from geomdl.

    Regularisation:
        The thin-wire approximation breaks down for field points inside
        the wire (|R| < wire_radius). Those points are set to zero —
        they have no physical meaning for the external field.
        min_dist = wire_radius prevents 1/|R|² from exploding.
    """
    ts        = np.linspace(0., 1., n_seg + 1)
    dt        = 1.0 / n_seg
    prefac    = MU0 * current / (4.0 * np.pi)
    B         = np.zeros((len(field_pts), 3))
    # Track minimum distance from each field point to any wire segment
    min_R_mag = np.full(len(field_pts), np.inf)

    for t in ts:
        ders = curve.derivatives(t, order=1)
        Ct   = np.array(ders[0])   # C(t)
        dCdt = np.array(ders[1])   # C'(t) — analytic NURBS tangent

        dl   = dCdt * dt

        R    = field_pts - Ct                                        # (M,3)
        Rmag = np.linalg.norm(R, axis=1)                            # (M,)
        min_R_mag = np.minimum(min_R_mag, Rmag)

        # Skip contribution for points too close to wire
        valid     = Rmag > wire_radius
        Rmag_safe = np.maximum(Rmag, wire_radius)
        Rhat      = R / Rmag_safe[:,None]

        B[valid] += prefac * np.cross(dl, Rhat[valid]) / Rmag_safe[valid,None]**2

    # Zero out field points inside the wire — physically undefined
    inside_wire     = min_R_mag < wire_radius
    B[inside_wire]  = 0.0
    n_inside        = inside_wire.sum()
    if n_inside > 0:
        print(f"  {n_inside} field points inside wire radius — set to zero")

    return B


# ═════════════════════════════════════════════════════════════════════════════
# 8.  CORE MAGNETIZATION ITERATION
#     Equations [5], [6], [7], [8]
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# 8.  CORE FIELD — SURFACE BOUND CURRENT, SOLVED AS LINEAR SYSTEM
#
#  For a linear isotropic core J_m = ∇×M = 0 inside → only surface bound
#  current K_s = M × n̂ matters.
#
#  Self-consistency at each surface sample point i:
#    B_i = B_coil(r_i) + (μ₀/4π) Σ_j [ (M_j × n̂_j)·dA × R̂_ij ] / R_ij²
#    M_i = χ · B_i
#
#  Substituting M = χB and treating M as the unknown vector:
#    M_i = χ · B_coil(r_i) + χ · (μ₀/4π) Σ_j T_ij · M_j
#    (I - χ·T) · m = χ · b_coil                              [LS]
#
#  where:
#    m      = [M_0x M_0y M_0z M_1x M_1y M_1z ...]  shape (3Ns,)
#    b_coil = [Bc_0x Bc_0y Bc_0z ...]               shape (3Ns,)
#    T_ij   = 3×3 block: field at r_i per unit M at r_j
#           = (μ₀/4π·dA) · [cross product operator (e_k × n̂_j)] × R̂_ij / R²
#
#  Solved with GMRES + diagonal preconditioner.
#  After solving → K_s_j = M_j × n̂_j
#  → evaluate B_core at all field grid points.
# ═════════════════════════════════════════════════════════════════════════════

def solve_core_linear(B_coil, field_pts, mu_r,
                      surf_pts, surf_normals, dA,
                      core_pmin, core_pmax,
                      tol, max_iter):
    """
    Solve surface bound current system (I - χT)·M = χ·B_coil via direct solve.
    All inputs in SI units (metres, Tesla).

    dA can be a scalar (uniform) or array of shape (Ns,) for per-point area weights.
    Supports lateral face + end caps — all faces included automatically.
    """
    from scipy.sparse.linalg import gmres
    from scipy.interpolate   import LinearNDInterpolator
    from scipy.spatial       import cKDTree

    # Correct susceptibility: M = χ·B where χ = (μᵣ-1)/(μ₀·μᵣ)
    chi  = (mu_r - 1.0) / (MU0 * mu_r)
    Ns       = len(surf_pts)
    prefac   = MU0 / (4.0 * np.pi)

    # dA: support both scalar and per-point array
    dA_arr = np.full(Ns, float(dA)) if np.isscalar(dA) else np.asarray(dA, dtype=float)
    min_dist = float(np.sqrt(dA_arr.mean())) * 0.5

    print(f"  Surface pts     : {Ns}")
    print(f"  χ = (μᵣ-1)/μ₀  : {chi:.4e}")

    # ── Interpolate B_coil onto surface points ────────────────────────────
    interp_x = LinearNDInterpolator(field_pts, B_coil[:,0])
    interp_y = LinearNDInterpolator(field_pts, B_coil[:,1])
    interp_z = LinearNDInterpolator(field_pts, B_coil[:,2])

    Bc_surf = np.column_stack([interp_x(surf_pts),
                               interp_y(surf_pts),
                               interp_z(surf_pts)])
    nan_mask = ~np.isfinite(Bc_surf).all(axis=1)
    if nan_mask.any():
        _, idx = cKDTree(field_pts).query(surf_pts[nan_mask])
        Bc_surf[nan_mask] = B_coil[idx]

    print(f"  B_coil at surf  : max={np.linalg.norm(Bc_surf,axis=1).max():.4e} T")

    # ── Build influence matrix T  (3Ns × 3Ns) ────────────────────────────
    # Packed as [M_0x, M_0y, M_0z, M_1x, M_1y, M_1z, ...]
    # T[3i:3i+3, 3j:3j+3] = 3×3 block: B at surf_pts[i] per unit M at surf_pts[j]
    #
    # For source j with M_j = e_k (unit vector along k):
    #   K_s = e_k × n̂_j              (surface current direction)
    #   dB at r_i = prefac·dA · (K_s × R̂_ij) / R²
    print(f"  Building {3*Ns}×{3*Ns} influence matrix T ...")

    T = np.zeros((3*Ns, 3*Ns))

    for j in range(Ns):
        R    = surf_pts - surf_pts[j]              # (Ns,3) vectors to all targets
        Rmag = np.linalg.norm(R, axis=1)           # (Ns,)
        Rmag[j] = np.inf                           # exclude self (handled separately)
        valid = Rmag > min_dist
        Rmag  = np.maximum(Rmag, min_dist)
        Rhat  = R / Rmag[:,None]                   # (Ns,3)
        c     = prefac * dA_arr / Rmag**2          # (Ns,) per-point dA

        for k in range(3):
            ek      = np.zeros(3); ek[k] = 1.0
            Ks_dir  = np.cross(ek, surf_normals[j])        # (3,) fixed direction
            # field at each target i from this source/direction:
            # dB_i = c_i * (Ks_dir × Rhat_i)
            dB = c[:,None] * np.cross(Ks_dir, Rhat)        # (Ns,3)
            dB[~valid] = 0.0

            # Place into T: column 3j+k, target block rows 3i:3i+3
            T[0::3, 3*j+k] = dB[:,0]   # x-component at all targets
            T[1::3, 3*j+k] = dB[:,1]   # y-component
            T[2::3, 3*j+k] = dB[:,2]   # z-component

    # ── Assemble system  (I - χT) · m = χ · b_coil ───────────────────────
    A     = np.eye(3*Ns) - chi * T
    b_rhs = chi * Bc_surf.ravel()          # (3Ns,)

    # Diagonal preconditioner: P = diag(A)^{-1}
    diag  = np.abs(np.diag(A))
    diag  = np.where(diag > 1e-30, diag, 1.0)
    P_inv = 1.0 / diag

    print(f"  A condition estimate: diag range [{diag.min():.3e}, {diag.max():.3e}]")
    print(f"  Solving (I - χT)·M = χ·B_coil via direct solve (numpy) ...")

    try:
        m_sol = np.linalg.solve(A, b_rhs)
        print(f"  Direct solve complete ✓")
        residual = np.linalg.norm(A @ m_sol - b_rhs) / np.linalg.norm(b_rhs)
        print(f"  Residual check    : {residual:.4e}")
        n_iters = [1]
        info    = 0
    except np.linalg.LinAlgError as e:
        print(f"  Direct solve failed ({e}) — falling back to GMRES ...")
        from scipy.sparse.linalg import LinearOperator
        M_op = LinearOperator((3*Ns, 3*Ns), matvec=lambda x: P_inv * x)
        n_iters = [0]
        def callback(res):
            n_iters[0] += 1
            if n_iters[0] % 20 == 0:
                print(f"    iter {n_iters[0]:4d}  residual={res:.4e}")
        m_sol, info = gmres(A, b_rhs, M=M_op, rtol=tol, maxiter=max_iter,
                            callback=callback, callback_type='pr_norm')
        if info == 0:
            print(f"  GMRES converged in {n_iters[0]} iterations ✓")
        else:
            print(f"  GMRES info={info} after {n_iters[0]} iters")

    M_surf = m_sol.reshape(Ns, 3)      # magnetisation at each surface point
    Mm     = np.linalg.norm(M_surf, axis=1)
    print(f"  |M| at surface  : max={Mm.max():.4e}  mean={Mm.mean():.4e} A/m")

    # ── Surface bound current K_s = M × n̂ ────────────────────────────────
    Ks_dA = np.cross(M_surf, surf_normals) * dA_arr[:,None]  # (Ns,3)  units: A·m

    # ── Evaluate B_core at ALL field grid points ──────────────────────────
    print(f"  Evaluating B_core at {len(field_pts)} grid points ...")
    B_core = np.zeros_like(field_pts)

    for j in range(Ns):
        R    = field_pts - surf_pts[j]                         # (Nf,3)
        Rmag = np.maximum(np.linalg.norm(R, axis=1), min_dist) # (Nf,)
        Rhat = R / Rmag[:,None]
        coef = prefac / Rmag**2                                 # (Nf,)
        B_core += coef[:,None] * np.cross(Ks_dA[j], Rhat)

    Bm_core = np.linalg.norm(B_core, axis=1)
    print(f"  B_core          : max={Bm_core.max():.4e} T  mean={Bm_core.mean():.4e} T")

    B_total = B_coil + B_core
    return B_total, n_iters[0], float(info)


# ═════════════════════════════════════════════════════════════════════════════
# 9.  FIELD GRID
# ═════════════════════════════════════════════════════════════════════════════

def make_grid(pmin, pmax, n=20, scale=1.2):
    c    = (pmin + pmax) / 2.
    half = (pmax - pmin) * scale / 2.
    xs   = np.linspace(c[0]-half[0], c[0]+half[0], n)
    ys   = np.linspace(c[1]-half[1], c[1]+half[1], n)
    zs   = np.linspace(c[2]-half[2], c[2]+half[2], n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    return X, Y, Z, np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


# ═════════════════════════════════════════════════════════════════════════════
# 10. OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════

def nurbs_curve_to_dict(curve):
    return dict(
        type       = "BSplineCurve",
        degree     = curve.degree,
        ctrlpts    = curve.ctrlpts,
        knotvector = curve.knotvector,
    )

def nurbs_surface_to_dict(surf):
    return dict(
        type           = "BSplineSurface",
        degree_u       = surf.degree_u,
        degree_v       = surf.degree_v,
        ctrlpts        = surf.ctrlpts,
        ctrlpts2d      = surf.ctrlpts2d,
        ctrlpts_size_u = surf.ctrlpts_size_u,
        ctrlpts_size_v = surf.ctrlpts_size_v,
        knotvector_u   = surf.knotvector_u,
        knotvector_v   = surf.knotvector_v,
    )

def save_outputs(coil_curve, core_surf, B, X, Y, Z, meta):
    np.savez("field_total.npz",
             Bx   = B[:,0].reshape(X.shape),
             By   = B[:,1].reshape(X.shape),
             Bz   = B[:,2].reshape(X.shape),
             Bmag = np.linalg.norm(B, axis=1).reshape(X.shape),
             X=X, Y=Y, Z=Z)

    with open("coil_nurbs.json", "w") as f:
        json.dump(nurbs_curve_to_dict(coil_curve), f, indent=2)

    if core_surf is not None:
        with open("core_nurbs.json", "w") as f:
            json.dump(nurbs_surface_to_dict(core_surf), f, indent=2)

    with open("metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n  Outputs saved:")
    print("    field_total.npz  — Bx,By,Bz,Bmag,X,Y,Z")
    print("    coil_nurbs.json  — control pts (n_c×3), knots, degree")
    if core_surf is not None:
        print("    core_nurbs.json  — control pts (n_u×n_v×3), knots, degree")
    print("    metadata.json    — all parameters + results")


def plot_results(centreline, X, Y, Z, B):
    Bmag = np.linalg.norm(B, axis=1).reshape(X.shape)
    kz   = Bmag.shape[2] // 2
    ky   = Bmag.shape[1] // 2

    # Clip colormap at 95th percentile to prevent near-wire spikes
    # from washing out the interior field structure
    vmax = float(np.percentile(Bmag, 95))
    vmin = 0.0

    fig  = plt.figure(figsize=(16, 4))

    ax1 = fig.add_subplot(141)
    ax1.plot(centreline[:,0], centreline[:,1], lw=0.8, color='steelblue')
    ax1.set_aspect('equal')
    ax1.set_title('Centreline XY')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')

    ax2 = fig.add_subplot(142)
    ax2.plot(centreline[:,0], centreline[:,2], lw=0.8, color='steelblue')
    ax2.set_title('Centreline XZ')
    ax2.set_xlabel('x'); ax2.set_ylabel('z')

    ax3 = fig.add_subplot(143)
    im1 = ax3.imshow(Bmag[:,:,kz].T, origin='lower',
                     extent=[X[0,0,0],X[-1,0,0],Y[0,0,0],Y[0,-1,0]],
                     aspect='auto', cmap='inferno',
                     vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=ax3, label='|B| (T)')
    ax3.set_title('|B| mid Z-slice (XY)  [clipped 95th pct]')
    ax3.set_xlabel('x'); ax3.set_ylabel('y')

    ax4 = fig.add_subplot(144)
    im2 = ax4.imshow(Bmag[:,ky,:].T, origin='lower',
                     extent=[X[0,0,0],X[-1,0,0],Z[0,0,0],Z[0,0,-1]],
                     aspect='auto', cmap='inferno',
                     vmin=vmin, vmax=vmax)
    plt.colorbar(im2, ax=ax4, label='|B| (T)')
    ax4.set_title('|B| mid Y-slice (XZ)  [clipped 95th pct]')
    ax4.set_xlabel('x'); ax4.set_ylabel('z')

    plt.tight_layout()
    plt.savefig('summary.png', dpi=150)
    print("    summary.png      — centreline + |B| field slices")


# ═════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="NURBS EM solver — NURBS parameters auto-detected from geometry.")
    ap.add_argument("--step",       type=str,   default=None)
    ap.add_argument("--current",    type=float, default=None)
    ap.add_argument("--frequency",  type=float, default=None)
    ap.add_argument("--mu_r",       type=float, default=None)
    ap.add_argument("--sigma_core", type=float, default=None)
    ap.add_argument("--grid_n",     type=int,   default=None)
    ap.add_argument("--tol",        type=float, default=None)
    ap.add_argument("--max_iter",   type=int,   default=None)
    args = ap.parse_args()
    args = prompt_inputs(args)

    I          = args.current
    freq       = args.frequency
    mu_r       = args.mu_r
    sigma_core = args.sigma_core

    # Skin depths
    delta_cu   = np.sqrt(1.0 / (np.pi * freq * MU0 * SIGMA_CU))
    delta_core = (np.sqrt(1.0 / (np.pi * freq * MU0 * mu_r * sigma_core))
                  if sigma_core > 0 else float('inf'))
    print(f"\n  Skin depth copper : {delta_cu*1e3:.4f} mm")
    print(f"  Skin depth core   : "
          f"{'%.4f mm' % (delta_core*1e3) if sigma_core > 0 else 'non-conductive'}")

    # ── Load STEP ─────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading {args.step} ...")
    shape  = load_step(args.step)
    solids = get_solids(shape)
    print(f"  {len(solids)} solid(s) found")
    coil_solid, core_solid = identify_solids(solids)
    pmin_all, pmax_all = bbox(shape)

    # ── Auto-analyse coil geometry ────────────────────────────────────────
    print(f"\n[2/6] Analysing coil geometry ...")
    coil_geo = analyse_coil_geometry(coil_solid)

    # ── Fit NURBS coil curve C(t) ─────────────────────────────────────────
    print(f"\n[3/6] Fitting coil NURBS curve C(t) ...")
    coil_curve, cl = fit_coil_nurbs(
        coil_geo['cl'],
        n_ctrl  = coil_geo['n_ctrl'],
        degree  = 3,
    )

    # ── Auto-analyse + fit NURBS core surface S(u,v) ──────────────────────
    core_surf      = None
    core_surf_pts  = None
    core_surf_nrm  = None
    core_geo       = None
    dA             = 1.0
    core_pmin      = None
    core_pmax      = None

    if core_solid is not None:
        print(f"\n[4/6] Analysing core geometry ...")
        core_geo = analyse_core_geometry(core_solid)

        print(f"\n[5/6] Fitting core NURBS surface S(u,v) + end caps ...")
        core_surf_pts, core_surf_nrm, dA, total_area = fit_core_nurbs(
            core_solid,
            n_su    = core_geo['n_su'],
            n_sv    = core_geo['n_sv'],
            n_cu    = core_geo['n_cu'],
            n_cv    = core_geo['n_cv'],
            degree  = 2,
        )
        # dA is now a per-point array (Ns,) — different for lateral vs end cap faces
        mean_dA = float(dA.mean()) if hasattr(dA, '__len__') else float(dA)
        print(f"  Surface samples : {len(core_surf_pts)}   mean dA = {mean_dA:.4e}")
        core_pmin, core_pmax = bbox(core_solid)
    else:
        print(f"\n[4/6] No core solid found — air core")
        print(f"[5/6] Skipping core surface fitting")

    # ── Field grid ────────────────────────────────────────────────────────
    print(f"\n[6/6] Building {args.grid_n}³ evaluation grid ...")
    # Grid centred on COIL bbox with 1.3x scale.
    # This captures: inside core, between core and coil, just outside coil.
    # Using the full shape bbox (pmin_all/pmax_all) was wrong for solenoids
    # where the bbox extends far beyond the coil, wasting resolution.
    coil_pmin, coil_pmax = bbox(coil_solid)
    X, Y, Z, fpts = make_grid(coil_pmin, coil_pmax, n=args.grid_n, scale=1.3)
    print(f"  {len(fpts)} field points  (grid spans coil + 30% margin)")

    # ── Unit conversion: mm → m ───────────────────────────────────────────
    # STEP file is in millimetres. All Biot-Savart distances must be in
    # metres for μ₀ = 4π×10⁻⁷ H/m to give correct results in Tesla.
    # We convert all geometry to metres here once, before any physics.
    MM2M = 1e-3
    print(f"\n  Converting all geometry mm → m (scale={MM2M})")

    # Field grid points → metres
    fpts_m = fpts * MM2M
    X_m    = X    * MM2M
    Y_m    = Y    * MM2M
    Z_m    = Z    * MM2M

    # Coil centreline control points → metres
    # geomdl requires set_ctrlpts() or direct list assignment via the setter
    ctrlpts_m = (np.array(coil_curve.ctrlpts) * MM2M).tolist()
    coil_curve.ctrlpts = ctrlpts_m
    # Verify the update took effect
    cp_check = np.array(coil_curve.ctrlpts)
    print(f"  Coil ctrl pts range after scaling: "
          f"X={cp_check[:,0].min():.4f}..{cp_check[:,0].max():.4f} m  "
          f"Z={cp_check[:,2].min():.4f}..{cp_check[:,2].max():.4f} m")

    # Wire radius → metres
    wire_radius_m = (coil_geo['wire_d'] / 2.0) * MM2M
    print(f"  Wire radius    : {wire_radius_m*1e3:.4f} mm = {wire_radius_m:.6f} m")

    # Core surface points and area → metres
    if core_surf_pts is not None:
        core_surf_pts_m = core_surf_pts * MM2M
        dA_m            = dA * MM2M**2
        core_pmin_m     = core_pmin * MM2M
        core_pmax_m     = core_pmax * MM2M
        dA_m_mean = float(dA_m.mean()) if hasattr(dA_m, '__len__') else float(dA_m)
        dA_m_len  = len(dA_m) if hasattr(dA_m, '__len__') else 1
        print(f"  Core dA        : mean={dA_m_mean:.4e} m2  ({dA_m_len} pts)")
    else:
        core_surf_pts_m = None
        dA_m            = dA * MM2M**2
        core_pmin_m     = None
        core_pmax_m     = None

    # ── Coil Biot-Savart ──────────────────────────────────────────────────
    print(f"\n  Computing B_coil via NURBS Biot-Savart "
          f"({coil_geo['n_seg']} steps) ...")
    B_coil = biot_savart_coil(
        coil_curve, I, fpts_m,
        n_seg       = coil_geo['n_seg'],
        wire_radius = wire_radius_m,
    )
    Bm = np.linalg.norm(B_coil, axis=1)
    print(f"  Coil only → max={Bm.max():.4e} T   mean={Bm.mean():.4e} T")
    print(f"  Coil only → finite values: {np.isfinite(B_coil).all()}")

    # ── Core field — linear system solve ─────────────────────────────────
    if core_surf_pts_m is not None and mu_r > 1.0:
        print(f"\n  Solving core surface bound current as linear system (GMRES) ...")
        print(f"  (Linear core → J_m=0 → surface K_s only, solved exactly)")
        B_total, iters, err = solve_core_linear(
            B_coil, fpts_m, mu_r,
            core_surf_pts_m, core_surf_nrm, dA_m,
            core_pmin_m, core_pmax_m,
            args.tol, args.max_iter,
        )
    else:
        B_total, iters, err = B_coil, 0, 0.0
        print(f"\n  Air core or mu_r=1 — skipping magnetization")

    # ── Results ───────────────────────────────────────────────────────────
    Bm_t = np.linalg.norm(B_total, axis=1)
    print(f"\n{'='*56}")
    print(f"  Results")
    print(f"{'='*56}")
    print(f"  max  |B| = {Bm_t.max():.6e} T")
    print(f"  min  |B| = {Bm_t.min():.6e} T")
    print(f"  mean |B| = {Bm_t.mean():.6e} T")
    print(f"  Iterations : {iters}   final err = {err:.2e}")

    meta = dict(
        step_file            = args.step,
        current_A            = I,
        frequency_Hz         = freq,
        mu_r                 = mu_r,
        sigma_core           = sigma_core,
        skin_depth_copper_mm = float(delta_cu * 1e3),
        skin_depth_core_mm   = float(delta_core*1e3) if sigma_core > 0 else None,
        B_max_T              = float(Bm_t.max()),
        B_min_T              = float(Bm_t.min()),
        B_mean_T             = float(Bm_t.mean()),
        iterations           = iters,
        final_error          = float(err),
        grid_n               = args.grid_n,
        # Auto-detected geometry params
        coil_wire_d          = float(coil_geo['wire_d']),
        coil_radius          = float(coil_geo['coil_r']),
        coil_height          = float(coil_geo['height']),
        coil_est_turns       = int(coil_geo['n_turns']),
        coil_n_seg           = int(coil_geo['n_seg']),
        nurbs_coil_nctrl     = len(coil_curve.ctrlpts),
        nurbs_coil_degree    = coil_curve.degree,
        nurbs_core_nctrl_u   = core_geo['n_cu'] if core_solid is not None else None,
        nurbs_core_nctrl_v   = core_geo['n_cv'] if core_solid is not None else None,
        nurbs_core_degree_u  = 2                if core_solid is not None else None,
        nurbs_core_degree_v  = 2                if core_solid is not None else None,
    )

    save_outputs(coil_curve, core_surf, B_total, X, Y, Z, meta)
    plot_results(cl, X, Y, Z, B_total)
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
