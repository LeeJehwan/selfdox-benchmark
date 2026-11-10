"""Score a submission against the GT.

    python -m selfdox.score --gt data/dev/gt.json --sub outputs --out results.json

Several runs of the same agent (repeat --sub) also report the score mean +- standard
deviation over runs and pass^k (images that passed in all k runs):

    python -m selfdox.score --gt data/dev/gt.json --sub run1 --sub run2 --sub run3
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import config as C
from .io import load_rgb
from .metrics import OUTCOMES, aggregate, image_outcome, multi_run, score_image
from .schema import load_gt, submission_format_errors


def invalid_predictions(sub_dir, gt):
    """({id: reason}, messages) for predictions that fail validation for one image (a box
    outside the photo, a duplicate id, a wrong size...). The official runs
    (tools/run_container.py) score those images 0, and so does this scorer."""
    from .schema import gt_sizes, submission_problems
    _, problems, _ = submission_problems(sub_dir, gt_sizes(gt))
    known = {im["id"] for im in gt["images"]}
    zero = {iid: msgs[0].split("): ", 1)[-1] for iid, msgs in problems.items() if iid in known}
    return zero, [m for msgs in problems.values() for m in msgs]


def score_submission(gt_path, sub_dir, zero=None):
    """Score every GT image. `zero` maps image ids to a reason: those images are scored
    as usual (so a GPS leak still shows) but get 0 points and that reason as hard fail
    (tools.run_container uses it for predictions that fail validation)."""
    gt_path, sub_dir = Path(gt_path), Path(sub_dir)
    gt = load_gt(gt_path)
    sub = json.loads((sub_dir / "predictions.json").read_text())
    preds = {}
    for p in sub.get("predictions", []):
        preds.setdefault(p.get("id"), p)
    results = []
    for im in gt["images"]:
        orig = load_rgb(gt_path.parent / im["file"])
        r = score_image(im, orig, preds.get(im["id"]), sub_dir)
        if zero and im["id"] in zero:
            r.score, r.passed = 0.0, False
            r.hard_fail = r.hard_fail or f"invalid_prediction:{zero[im['id']]}"
            r.outcome = image_outcome(r)
        results.append(r)
    return gt, sub, aggregate(results), results


def _fmt(v):
    return "  -  " if v is None else f"{v:5.3f}"


def _ci(ci, d=3):
    return "  -  " if ci is None else f"[{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def print_summary(agent, agg):
    p = agg["parts"]
    print(f"SelfDox scorer {C.VERSION}  agent={agent}  images={agg['n_images']}")
    print(f"  SCORE {agg['score']:6.2f}   false-safe rate {agg['false_safe_rate']:.3f} "
          f"({agg['n_safe_verdicts']} 'safe' verdicts)   pass rate {agg['pass_rate']:.3f}")
    print("  L " + _fmt(p["L"]) + "  E " + _fmt(p["E"]) + "  PU " + _fmt(p["PU"])
          + "  R " + _fmt(p["R"]) + "  V " + _fmt(p["V"]))
    for cat, row in agg["per_category"].items():
        print(f"  {cat:<12} n={row['n']:<3} score {row['score']:6.2f}")
    if "outcomes" in agg:
        n = agg["n_images"] or 1
        o = agg["outcomes"]
        print("  outcomes: " + "  ".join(f"{k.replace('_', ' ')} {o[k]['n']} ({o[k]['n'] / n:.0%})"
                                         for k in OUTCOMES))
        ci = agg["ci95"]
        print(f"  95% CI: score {_ci(ci['score'], 1)}  pass rate {_ci(ci['pass_rate'])}  "
              f"cleared {_ci(ci['cleared_rate'])}  leaks {_ci(ci['leak_rate'])}  "
              f"(leaks among 'safe' verdicts {agg['safe_leak_rate']:.3f})")
        for g, row in agg["per_group"].items():
            print(f"  {g:<12} n={row['n']:<3} score {row['score']:6.2f}  pass {row['pass_rate']:.2f}  "
                  + " ".join(f"{k.split('_')[0]} {v}" for k, v in row["outcomes"].items()))
    if agg["hard_fails"]:
        print("  hard fails (image scores 0): " + ", ".join(f"{k}={v}" for k, v in
                                                          agg["hard_fails"].items()))
        for k, ids in (agg.get("hard_fail_ids") or {}).items():
            print(f"    {k}: {' '.join(ids[:12])}{' ...' if len(ids) > 12 else ''}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a SelfDox submission.")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sub", required=True, action="append",
                    help="folder with predictions.json and images/ (repeat for several runs)")
    ap.add_argument("--out", help="write full per-image results as JSON")
    args = ap.parse_args(argv)
    for sub_dir in args.sub:
        _, errors = submission_format_errors(sub_dir)
        if errors:
            print(f"{sub_dir}: submission does not match the format; run selfdox.validate first:")
            print("\n".join("  " + e for e in errors))
            return 2
    runs = []
    gt_doc = load_gt(args.gt)
    for sub_dir in args.sub:
        zero, msgs = invalid_predictions(sub_dir, gt_doc)
        for m in msgs:
            print(f"WARNING {m}" + (" -> image scored 0" if m.split(" (")[1].split(")")[0] in zero
                                    else ""))
        gt, sub, agg, results = score_submission(args.gt, sub_dir, zero=zero)
        print_summary(sub.get("agent_name"), agg)
        runs.append({"sub": str(sub_dir), "agent_name": sub.get("agent_name"), "summary": agg,
                     "images": results})
    multi = multi_run([r["images"] for r in runs]) if len(runs) > 1 else None
    if multi:
        print(f"{multi['k']} runs: score {multi['score_mean']:.2f} +- {multi['score_std']:.2f}  "
              f"pass@1 {multi['pass_at_1']:.3f}  pass^{multi['k']} {multi['pass_hat_k']:.3f}")
    if args.out:
        head = {"scorer_version": C.VERSION, "benchmark_version": gt["benchmark_version"]}
        if multi is None:
            r = runs[0]
            body = {"agent_name": r["agent_name"], "summary": r["summary"],
                    "images": [asdict(x) for x in r["images"]]}
        else:
            body = {"multi_run": multi,
                    "runs": [{"sub": r["sub"], "agent_name": r["agent_name"],
                              "summary": r["summary"],
                              "images": [asdict(x) for x in r["images"]]} for r in runs]}
        Path(args.out).write_text(json.dumps({**head, **body}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
