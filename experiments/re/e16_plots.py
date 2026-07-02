"""Thin CLI wrapper around the canonical E16 pipeline.

For each grid with both
  - ``<input-dir>/MoneeResilienceExperiment-<grid>/network.p``  (post-solve net)
  - ``<shed-dir>/single_removal_shed_<grid>.csv``               (shed ground truth)

this script:

  1. rebuilds the same artefact ``cp_cn_evaluation.evaluate`` would assemble
     for that grid (impact_df ← create_impact_df, df_eval ← cp_metric_vs_actual_impact),
  2. delegates the ρ table + per-scenario merged CSVs to
     ``cmres_eval.experiment_e16_single_removal_validation``,
  3. delegates the plotting to
     ``cmres_eval_plots.plot_e16_single_removal`` so the figures stay
     consistent with the rest of the dissertation.

Outputs land in ``<output-dir>`` (default ``data/out/cmres``):
  - ``E16_metric_vs_shed.csv``        ρ per (scenario, metric)
  - ``E16_shed_vs_mc_ceiling.csv``    ρ between analytical shed and MC actual
  - ``E16_<scenario>_merged.csv``     raw merged frame per scenario
  - ``E16_single_removal.html`` (+ ``single/*.pdf``)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List

import dill
import pandas

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "re"))

from cmres_eval import ScenarioArtefacts, experiment_e16_single_removal_validation  # noqa: E402
from cmres_eval_plots import plot_e16_single_removal  # noqa: E402
from cp_cn_evaluation import (  # noqa: E402
    _scenario_density_distribution,
    create_impact_df,
    create_metrics_df,
    cp_metric_vs_actual_impact,
    extend_impact_df,
)


DEFAULT_INPUT_DIR = Path("data/res")
DEFAULT_SHED_DIR = Path("data/out/single_removal_shed")
DEFAULT_OUT_DIR = Path("data/out/cmres")


def _load_grid_dfs(grid: str, input_dir: Path):
    """Replicate the per-grid slice of ``cp_cn_evaluation.load_dfs`` so we can
    rebuild ``impact_df`` and the artefact for a single grid without scanning
    every sibling experiment folder."""
    exp = input_dir / f"MoneeResilienceExperiment-{grid}"
    perf_path = exp / "performance.csv"
    fail_path = exp / "failure.csv"
    net_path = exp / "network.p"
    for p in (perf_path, fail_path, net_path):
        if not p.exists():
            raise FileNotFoundError(p)

    perf_df = pandas.read_csv(perf_path)
    perf_df["experiment"] = str(exp)
    perf_df["network_type"] = grid

    fail_df = pandas.read_csv(fail_path)
    fail_df["experiment"] = str(exp)
    fail_df["network_type"] = grid

    with open(net_path, "rb") as f:
        monee_net = dill.load(f)

    return perf_df, fail_df, monee_net


def run_one(grid: str, input_dir: Path, shed_dir: Path, output_dir: Path):
    print(f"\n=== E16 :: {grid} ===")
    perf_df, fail_df, monee_net = _load_grid_dfs(grid, input_dir)
    print(f"  building metrics_df + impact_df …")
    metrics_df = create_metrics_df(monee_net, grid)
    impact_df = create_impact_df(perf_df, fail_df, metrics_df)
    impact_df = extend_impact_df({grid: monee_net}, metrics_df, impact_df)
    impact_df_nt = impact_df[impact_df["network_type"] == grid]

    print(f"  running cp_metric_vs_actual_impact (PTDF + all-component metric) …")
    df_eval = cp_metric_vs_actual_impact(monee_net, impact_df_nt, grid)
    if df_eval is None or len(df_eval) == 0:
        print(f"[E16:{grid}] df_eval empty; skipping")
        return None

    density, distribution = _scenario_density_distribution(grid)
    artefact = ScenarioArtefacts(
        label=grid,
        df_eval=df_eval,
        monee_net=monee_net,
        mc_npz_path=None,
        density=density,
        distribution=distribution,
    )
    return artefact


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("grids", nargs="*",
                    help="grid names; default = every shed CSV in --shed-dir")
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                    help="dir with MoneeResilienceExperiment-<grid>/ subfolders")
    ap.add_argument("--shed-dir", type=Path, default=DEFAULT_SHED_DIR,
                    help="dir containing single_removal_shed_<grid>.csv files")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="dir where E16_*.csv/html land")
    args = ap.parse_args()

    grids: List[str] = args.grids or [
        p.stem.replace("single_removal_shed_", "")
        for p in sorted(args.shed_dir.glob("single_removal_shed_*.csv"))
        if "_shard_" not in p.stem
    ]
    if not grids:
        print(f"no shed CSVs in {args.shed_dir}", file=sys.stderr)
        return 1
    print(f"E16 for grids: {grids}")

    artefacts: List[ScenarioArtefacts] = []
    for g in grids:
        try:
            art = run_one(g, args.input_dir, args.shed_dir, args.output_dir)
            if art is not None:
                artefacts.append(art)
        except Exception as e:
            print(f"[E16:{g}] artefact build FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    if not artefacts:
        print("no usable artefacts; nothing to do", file=sys.stderr)
        return 1

    print(f"\n→ experiment_e16_single_removal_validation on {len(artefacts)} grid(s)")
    experiment_e16_single_removal_validation(
        artefacts, args.output_dir, shed_dir=args.shed_dir,
    )
    print(f"→ plot_e16_single_removal (cmres_eval_plots)")
    html_path = plot_e16_single_removal(args.output_dir, args.output_dir)
    print(f"  report → {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
