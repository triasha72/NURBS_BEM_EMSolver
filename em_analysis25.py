"""
em_analysis25.py  —  Batch NURBS EM Solver for 25 solenoids
=============================================================
Runs the full EM analysis on solenoid_01.step through solenoid_25.step.
Outputs for each solenoid are saved in their own folder:

    C:\\temp\\solenoids_output\\solenoid_01\\
        field_total.npz
        coil_nurbs.json
        core_nurbs.json
        metadata.json
        summary.png

    C:\\temp\\solenoids_output\\solenoid_02\\
        ...

Physics parameters (current, mu_r etc.) are prompted once at the start
and applied to all 25 solenoids.

Usage:
    python em_analysis25.py

Or with arguments to skip prompts:
    python em_analysis25.py --current 1.0 --frequency 100 --mu_r 100
                            --sigma_core 0 --grid_n 20
"""

import os
import sys
import json
import time
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

SOLENOID_DIR = r"C:\temp\solenoids_output"   # where solenoid_XX.step files live
OUTPUT_DIR   = r"C:\temp\solenoids_output"   # where per-solenoid folders are created
N_SOLENOIDS  = 25


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT PHYSICS INPUTS ONCE
# ═════════════════════════════════════════════════════════════════════════════

def prompt_physics():
    """Ask the user for physics parameters once, applied to all solenoids."""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--current",    type=float, default=None)
    ap.add_argument("--frequency",  type=float, default=None)
    ap.add_argument("--mu_r",       type=float, default=None)
    ap.add_argument("--sigma_core", type=float, default=None)
    ap.add_argument("--grid_n",     type=int,   default=None)
    args, _ = ap.parse_known_args()

    print("\n" + "="*56)
    print("  Batch EM Analysis — 25 Solenoids")
    print("  Physics inputs (applied to all solenoids)")
    print("="*56)

    def ask(prompt, default, cast):
        if getattr(args, prompt.split()[0].lower().replace(' ','_'), None) is not None:
            return getattr(args, prompt.split()[0].lower().replace(' ','_'))
        val = input(f"  {prompt} [{default}]: ").strip()
        return cast(val) if val else cast(default)

    I          = ask("Wire current I [A]",            "1",    float)
    freq       = ask("Operating frequency f [Hz]",    "100",  float)
    mu_r       = ask("Relative permeability mu_r",    "100",  float)
    sigma_core = ask("Core conductivity sigma_core [S/m]\n"
                     "    (ferrite ~0.01, iron powder ~1000, air=0)",
                     "0", float)
    grid_n     = ask("Grid points per axis N",        "20",   int)

    return dict(
        current    = I,
        frequency  = freq,
        mu_r       = mu_r,
        sigma_core = sigma_core,
        grid_n     = grid_n,
    )


# ═════════════════════════════════════════════════════════════════════════════
# RUN ONE SOLENOID  — subprocess call to em_analysis.py
# ═════════════════════════════════════════════════════════════════════════════

def run_one(name, step_path, out_dir, physics):
    """
    Calls em_analysis.py as a subprocess with all args on the command line.
    Captures stdout/stderr, moves outputs into out_dir.
    Returns metadata dict parsed from metadata.json, or raises on failure.
    """
    os.makedirs(out_dir, exist_ok=True)

    em_script = os.path.join(THIS_DIR, "em_analysis.py")

    cmd = [
        sys.executable, em_script,
        "--step",       step_path,
        "--current",    str(physics['current']),
        "--frequency",  str(physics['frequency']),
        "--mu_r",       str(physics['mu_r']),
        "--sigma_core", str(physics['sigma_core']),
        "--grid_n",     str(physics['grid_n']),
        "--tol",        "1e-6",
        "--max_iter",   "1000",
    ]

    # Run from out_dir so all output files land there directly
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd,
        cwd    = out_dir,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text   = True,
        env    = env,
        encoding = "utf-8",
        errors   = "replace",
    )

    # Print output with indentation
    for line in result.stdout.splitlines():
        print(f"    {line}")

    if result.returncode != 0:
        raise RuntimeError(f"em_analysis.py exited with code {result.returncode}")

    # Read metadata.json written by em_analysis.py
    meta_path = os.path.join(out_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise RuntimeError("metadata.json not found after run")

    with open(meta_path) as f:
        meta = json.load(f)

    return meta


# ═════════════════════════════════════════════════════════════════════════════
# BATCH LOOP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    physics = prompt_physics()

    print(f"\n{'='*56}")
    print(f"  Starting batch run — {N_SOLENOIDS} solenoids")
    print(f"  current={physics['current']} A  "
          f"mu_r={physics['mu_r']}  "
          f"grid_n={physics['grid_n']}")
    print(f"{'='*56}\n")

    results = []
    t_start = time.time()

    for i in range(1, N_SOLENOIDS + 1):
        name      = f"solenoid_{i:02d}"
        step_path = os.path.join(SOLENOID_DIR, f"{name}.step")
        out_dir   = os.path.join(OUTPUT_DIR,   name)

        print(f"\n{'─'*56}")
        print(f"  [{i:02d}/{N_SOLENOIDS}]  {name}")
        print(f"{'─'*56}")

        if not os.path.isfile(step_path):
            print(f"  ✗  STEP file not found: {step_path} — skipping")
            results.append((name, "missing", None))
            continue

        t0 = time.time()
        try:
            meta    = run_one(name, step_path, out_dir, physics)
            elapsed = time.time() - t0
            print(f"  ✓  Done in {elapsed:.1f}s")
            results.append((name, "ok", meta))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ✗  FAILED after {elapsed:.1f}s: {e}")
            results.append((name, "error", str(e)))

    total_time = time.time() - t_start

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"  Batch complete — {total_time:.1f}s total")
    print(f"{'='*56}")
    print(f"  {'Solenoid':<14} {'Status':<10} {'B_max (T)':<14} {'B_mean (T)'}")
    print(f"  {'-'*52}")

    ok_count  = 0
    err_count = 0
    mis_count = 0

    for name, status, payload in results:
        if status == "ok":
            ok_count += 1
            print(f"  {name:<14} {'✓ ok':<10} "
                  f"{payload['B_max_T']:<14.4e} {payload['B_mean_T']:.4e}")
        elif status == "missing":
            mis_count += 1
            print(f"  {name:<14} {'- missing':<10}")
        else:
            err_count += 1
            print(f"  {name:<14} {'✗ error':<10} {str(payload)[:40]}")

    print(f"\n  OK: {ok_count}   Errors: {err_count}   Missing: {mis_count}")

    # ── Save batch summary JSON ───────────────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "batch_summary.json")
    summary_data = []
    for name, status, payload in results:
        entry = dict(solenoid=name, status=status)
        if status == "ok":
            entry.update(payload)
        elif status == "error":
            entry["error"] = str(payload)
        summary_data.append(entry)

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n  Batch summary → {summary_path}")
    print(f"  Done.")


if __name__ == "__main__":
    main()
