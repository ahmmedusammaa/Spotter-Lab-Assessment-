"""End-to-end pipeline.

    python run_pipeline.py --data-dir data

Steps: EDA figures -> missing-data report -> time-based validation (strategies + baseline)
       -> final model -> validation_predictions.csv + filled data/december_chart_inputs.csv
Everything measured is written to reports/results.json (used for the written report).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import freight as fr  # noqa: E402

TEAL = "#064A56"
FOLDS = [  # expanding window: train on everything before `start`, test on [start, end)
    ("2025-05-01", "2025-07-01"),
    ("2025-07-01", "2025-09-01"),
    ("2025-09-01", "2025-11-01"),
]


def eda_figures(tr: pd.DataFrame, out: Path) -> dict:
    tr = tr.copy()
    tr["rpm"] = tr[fr.TARGET] / tr["distance"]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].hist(tr[fr.TARGET], bins=60, color=TEAL); ax[0, 0].set_title("posted_rate (right-skewed)")
    ax[0, 1].hist(np.log(tr[fr.TARGET]), bins=60, color=TEAL); ax[0, 1].set_title("log(posted_rate)")
    ax[1, 0].scatter(tr["distance"], tr[fr.TARGET], s=2, alpha=0.15, color=TEAL)
    ax[1, 0].set_title("Distance vs rate"); ax[1, 0].set_xlabel("distance (miles)")
    order = ["Dry Van", "Flatbed", "Reefer"]
    ax[1, 1].boxplot([tr.loc[tr.equipment == e, "rpm"].clip(upper=8) for e in order], showfliers=False)
    ax[1, 1].set_xticks([1, 2, 3], order)
    ax[1, 1].set_title("Rate per mile by equipment")
    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "eda_overview.png", dpi=150); plt.close(fig)

    month = tr["date"].dt.to_period("M").astype(str)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    tr.groupby(month)["market_index"].mean().plot(ax=ax[0], marker="o", color=TEAL, title="Mean market_index by month")
    tr.groupby(month)["rpm"].median().plot(ax=ax[1], marker="o", color=TEAL, title="Median rate per mile by month")
    for a in ax:
        a.set_xlabel(""); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "eda_time.png", dpi=150); plt.close(fig)

    return {
        "rpm_median_by_equipment": tr.groupby("equipment")["rpm"].median().round(3).to_dict(),
        "corr_with_rate": tr[["distance", "weight", "market_index", "quote_signal", fr.TARGET]].corr()[fr.TARGET].round(3).to_dict(),
        "rate_describe": tr[fr.TARGET].describe().round(2).to_dict(),
        "n_rpm_gt_10": int((tr["rpm"] > 10).sum()),
    }


def missing_report(tr, va, out: Path) -> dict:
    rep = {
        "train_rows": len(tr), "valid_rows": len(va),
        "train_missing": {c: int(tr[c].isna().sum()) for c in ["weight", "market_index"]},
        "valid_missing": {c: int(va[c].isna().sum()) for c in ["weight", "market_index"]},
        "train_negative_weight": int((tr["weight"] < 0).sum()),
        "valid_negative_weight": int((va["weight"] < 0).sum()),
        "any_other_missing_train": {c: int(v) for c, v in tr.isna().sum().items() if v and c not in ("weight", "market_index")},
        "duplicate_load_id": int(tr["load_id"].duplicated().sum()),
        "train_dates": [str(tr.date.min().date()), str(tr.date.max().date())],
        "valid_dates": [str(va.date.min().date()), str(va.date.max().date())],
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    labels = ["weight\nmissing", "weight\nnegative", "market_index\nmissing"]
    vals = [rep["train_missing"]["weight"], rep["train_negative_weight"], rep["train_missing"]["market_index"]]
    ax.bar(labels, vals, color=TEAL)
    for i, v in enumerate(vals):
        ax.text(i, v + 5, f"{v} ({v / len(tr) * 100:.2f}%)", ha="center", fontsize=9)
    ax.set_title("Data problems in train_test.csv"); ax.set_ylabel("rows"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "missing_data.png", dpi=150); plt.close(fig)
    return rep


# -------------------------------------------------------------------- baselines
WEIGHT_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]   # candidate weights on boosting in the blend


def validate(T: pd.DataFrame, out: Path) -> dict:
    """Three expanding-window time folds: missing-data strategies, baselines, blend weight."""
    res = {"folds": []}
    per_fold_logs = []          # (y, boost_log, linear_log, B) per fold, to sweep the blend weight
    for start, end in FOLDS:
        A = T[T["date"] < start]; B = T[(T["date"] >= start) & (T["date"] < end)]
        fold = {"train_end": start, "test": f"{start} to {end}", "n_train": len(A), "n_test": len(B)}
        a, b = fr.fill(A, A, "grouped_median"), fr.fill(B, A, "grouped_median")
        y = b[fr.TARGET].to_numpy()

        # --- missing-data strategies (boosting only)
        blend = fr.BlendModel().fit(a, a[fr.TARGET])                 # contains the grouped-median boosting model
        lb, ll = blend.boost_log(b), blend.linear_log(b)
        fold["grouped_median"] = fr.metrics(y, np.exp(lb)); fold["grouped_median"]["n_train"] = len(a)
        g_tr, g_te = fr.fill(A, A, "global_median"), fr.fill(B, A, "global_median")
        fold["global_median"] = fr.metrics(y, fr.predict(fr.fit(g_tr, g_tr[fr.TARGET]), g_te)); fold["global_median"]["n_train"] = len(A)
        d_tr = A.dropna(subset=["weight", "market_index"])             # delete rows from TRAINING only
        fold["drop_rows"] = fr.metrics(y, fr.predict(fr.fit(d_tr, d_tr[fr.TARGET]), b)); fold["drop_rows"]["n_train"] = len(d_tr)

        # --- loss ablation: ordinary squared-error boosting
        fold["boost_squared_loss"] = fr.metrics(y, fr.predict(fr.fit(a, a[fr.TARGET], loss="squared_error"), b))

        # --- baselines
        fold["linear_baseline"] = fr.metrics(y, np.exp(ll))
        rpm = (a[fr.TARGET] / a["distance"]).groupby(a["equip"]).median()
        fold["naive_rpm"] = fr.metrics(y, b["distance"] * b["equip"].map(rpm))
        fold["blend_by_weight"] = {str(w): fr.metrics(y, np.exp(w * lb + (1 - w) * ll)) for w in WEIGHT_GRID}
        res["folds"].append(fold); per_fold_logs.append((y, lb, ll, b))
        print(f"  fold {start}: boosting {fold['grouped_median']['MAPE_%']:.2f}% | linear {fold['linear_baseline']['MAPE_%']:.2f}%", flush=True)

    avg = lambda key: {k: round(float(np.mean([f[key][k] for f in res["folds"]])), 2) for k in ("MAE", "RMSE", "MAPE_%")}
    res["strategies"] = {s: avg(s) for s in ("drop_rows", "global_median", "grouped_median")}
    res["boost_squared_loss"] = avg("boost_squared_loss")
    res["baseline"] = {"linear_ridge_distance_equipment": avg("linear_baseline"), "naive_rate_per_mile": avg("naive_rpm")}
    sweep = {w: {k: round(float(np.mean([f["blend_by_weight"][w][k] for f in res["folds"]])), 2) for k in ("MAE", "RMSE", "MAPE_%")}
             for w in map(str, WEIGHT_GRID)}
    res["weight_sweep"] = sweep
    # choose on the unrounded average MAPE (ties at 2 decimals would otherwise depend on grid order)
    best_w = min(WEIGHT_GRID, key=lambda w: np.mean([f["blend_by_weight"][str(w)]["MAPE_%"] for f in res["folds"]]))
    res["best_blend_weight"] = best_w
    res["blend"] = sweep[str(best_w)]
    res["blend_folds"] = [f["blend_by_weight"][str(best_w)] for f in res["folds"]]

    # ---- diagnostics on the last (Sep-Oct) hold-out for the chosen blend
    y, lb, ll, B = per_fold_logs[-1]
    B = B.copy(); B["pred"] = np.exp(best_w * lb + (1 - best_w) * ll)
    B["ape"] = (B["pred"] - B[fr.TARGET]).abs() / B[fr.TARGET] * 100
    B["dist_bin"] = pd.cut(B["distance"], [0, 300, 700, 1200, 2000, 5000])
    res["holdout_by_distance"] = {str(k): round(float(v), 2) for k, v in B.groupby("dist_bin", observed=True)["ape"].mean().items()}
    res["holdout_by_equipment"] = B.groupby("equipment")["ape"].mean().round(2).to_dict()
    res["holdout_error_summary"] = {"median_ape": round(float(B["ape"].median()), 2), "mean_ape": round(float(B["ape"].mean()), 2),
                                    "share_gt_25pct": round(float((B["ape"] > 25).mean() * 100), 2)}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].scatter(B[fr.TARGET], B["pred"], s=3, alpha=0.2, color=TEAL)
    lim = [0, float(B[fr.TARGET].quantile(0.999))]; ax[0].plot(lim, lim, "r--", lw=1)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].set_xlabel("actual rate ($)"); ax[0].set_ylabel("predicted rate ($)")
    ax[0].set_title("Sep-Oct hold-out: predicted vs actual")
    B.groupby("dist_bin", observed=True)["ape"].mean().plot.bar(ax=ax[1], color=TEAL, rot=0)
    ax[1].set_title("Mean abs % error by distance (miles)"); ax[1].set_xlabel(""); ax[1].set_ylabel("%")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "holdout_diagnostics.png", dpi=150); plt.close(fig)

    # ---- permutation importance (boosting part, last fold): rise in log-MAE when a column is shuffled
    A = T[T["date"] < FOLDS[-1][0]]; a = fr.fill(A, A, "grouped_median")
    model = fr.fit(a, a[fr.TARGET]); yl = np.log(B[fr.TARGET].to_numpy()); Xh = fr.fill(B, A, "grouped_median")
    base = np.abs(model.predict(Xh[fr.FEATURES].to_numpy(float)) - yl).mean(); rng = np.random.default_rng(0); imp = {}
    for col in fr.FEATURES:
        vals = []
        for _ in range(2):
            Xp = Xh[fr.FEATURES].to_numpy(float).copy(); Xp[:, fr.FEATURES.index(col)] = rng.permutation(Xp[:, fr.FEATURES.index(col)])
            vals.append(np.abs(model.predict(Xp) - yl).mean() - base)
        imp[col] = round(float(np.mean(vals)), 4)
    res["permutation_importance"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:8])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--predictions-out", default="validation_predictions.csv")
    ap.add_argument("--december-out", default=None, help="default: <data-dir>/december_chart_inputs.csv (filled in place)")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    figs, reps = Path(args.figures_dir), Path(args.reports_dir)
    figs.mkdir(exist_ok=True, parents=True); reps.mkdir(exist_ok=True, parents=True)
    tr, va, template, dec = fr.load_data(args.data_dir)
    cats = fr.make_category_map(tr, va)

    results = {"eda": eda_figures(tr, figs), "missing": missing_report(tr, va, figs)}
    T = fr.prep(tr, cats)
    results["validation"] = validate(T, figs)

    # ---- final model: all labelled data, grouped-median fill
    T2 = fr.fill(T, T, "grouped_median")
    model = fr.BlendModel(w_boost=results["validation"]["best_blend_weight"]).fit(T2, T2[fr.TARGET])

    V2 = fr.fill(fr.prep(va, cats), T, "grouped_median")
    assert (template["load_id"].to_numpy() == va["load_id"].to_numpy()).all(), "template order differs from validation.csv"
    out = template.copy(); out["predicted_rate"] = model.predict(V2).round(2)
    out.to_csv(args.predictions_out, index=False)

    D = fr.fill(fr.prep(fr.build_december_features(dec, tr, va), cats), T, "grouped_median")
    dec_out = dec.copy(); dec_out["predicted_rate"] = model.predict(D).round(2)
    dec_file = "december-chart-inputs.csv" if (Path(args.data_dir) / "december-chart-inputs.csv").exists() else "december_chart_inputs.csv"
    dec_path = Path(args.december_out or Path(args.data_dir) / dec_file)
    dec_out.to_csv(dec_path, index=False)

    results["final"] = {
        "valid_pred_describe": out["predicted_rate"].describe().round(2).to_dict(),
        "december_min": float(dec_out["predicted_rate"].min()), "december_max": float(dec_out["predicted_rate"].max()),
        "december_mean": round(float(dec_out["predicted_rate"].mean()), 2),
    }
    (reps / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print("missing-data strategies:", json.dumps(results["validation"]["strategies"]))
    print("blend weight on boosting:", results["validation"]["best_blend_weight"])
    print("blend:", json.dumps(results["validation"]["blend"]))
    print("baselines:", json.dumps(results["validation"]["baseline"]))
    print(f"wrote {args.predictions_out} and {dec_path}")


if __name__ == "__main__":
    main()
