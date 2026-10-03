"""Shared helpers for the freight rate challenge: cleaning, missing-value filling, model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "posted_rate"
CAT_COLS = ["equipment", "pickup", "delivery"]

FEATURES = [
    "distance", "weight", "market_index", "quote_signal",
    "equip", "pickup_c", "delivery_c",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "dow", "mi_missing", "w_missing", "weight_bad",
]


# --------------------------------------------------------------------------- data
def _find_file(d: Path, names: list[str]) -> Path:
    for name in names:
        p = d / name
        if p.exists():
            return p
    return d / names[0]


def load_data(data_dir: str | Path):
    """Read the four challenge files from data_dir."""
    d = Path(data_dir)
    train_path = _find_file(d, ["train-test.csv", "train_test.csv"])
    valid_path = _find_file(d, ["validation.csv"])
    december_path = _find_file(d, ["december-chart-inputs.csv", "december_chart_inputs.csv"])
    template_path = _find_file(d, ["validation-predictions-template.csv", "validation_predictions_template.csv"])

    train = pd.read_csv(train_path, parse_dates=["date"])
    valid = pd.read_csv(valid_path, parse_dates=["date"])
    december = pd.read_csv(december_path)

    if template_path.exists():
        template = pd.read_csv(template_path)
        if "load_id" not in template.columns or len(template) != len(valid):
            template = pd.DataFrame({"load_id": valid["load_id"]})
    else:
        template = pd.DataFrame({"load_id": valid["load_id"]})

    return train, valid, template, december


def make_category_map(*frames: pd.DataFrame) -> dict[str, list[str]]:
    """Fixed category lists so integer codes are identical in every file."""
    return {c: sorted(set().union(*[set(f[c].dropna()) for f in frames if c in f])) for c in CAT_COLS}


def prep(df: pd.DataFrame, cats: dict[str, list[str]]) -> pd.DataFrame:
    """Clean obvious errors and build features. Does NOT fill missing values."""
    d = df.copy()
    d["weight_bad"] = (d["weight"] < 0).astype(int)      # negative weight = sign error
    d["weight"] = d["weight"].abs()
    d["mi_missing"] = d["market_index"].isna().astype(int)
    d["w_missing"] = d["weight"].isna().astype(int)
    d["month"] = d["date"].dt.month
    d["dow"] = d["date"].dt.dayofweek
    d["equip"] = pd.Categorical(d["equipment"], categories=cats["equipment"]).codes
    d["pickup_c"] = pd.Categorical(d["pickup"], categories=cats["pickup"]).codes
    d["delivery_c"] = pd.Categorical(d["delivery"], categories=cats["delivery"]).codes
    return d


def fill(d: pd.DataFrame, ref: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Fill missing weight / market_index.

    ref is the data the medians are learned from (training part only -> no leakage).
    strategies: 'global_median' | 'grouped_median'
      grouped_median: market_index by calendar month (seasonal), weight by equipment type.
    """
    d = d.copy()
    if strategy == "global_median":
        for c in ["weight", "market_index"]:
            d[c] = d[c].fillna(ref[c].median())
    elif strategy == "grouped_median":
        d["market_index"] = d["market_index"].fillna(d["month"].map(ref.groupby("month")["market_index"].median()))
        d["weight"] = d["weight"].fillna(d["equip"].map(ref.groupby("equip")["weight"].median()))
        # safety net if a group had no data
        for c in ["weight", "market_index"]:
            d[c] = d[c].fillna(ref[c].median())
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return d


# ------------------------------------------------------------------------- model
# Only numpy / pandas / matplotlib are available, so the models are implemented here:
#   * GradientBoosting : histogram-based gradient boosted trees (like LightGBM / sklearn's
#                        HistGradientBoostingRegressor, but a small plain-numpy version)
#   * RidgeRegression  : closed-form ridge
#   * BlendModel       : weighted average (in log space) of the two
class GradientBoosting:
    """Leaf-wise histogram gradient boosting for regression.

    loss='absolute_error' (default): robust LAD boosting - trees are fit to sign(residual) and each
        leaf value is the median residual in the leaf (line search), so extreme rows have little pull.
    loss='squared_error': ordinary least-squares boosting (leaf value = mean residual).
    """

    def __init__(self, loss="absolute_error", n_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                 min_samples_leaf=20, max_bins=255):
        if loss not in ("absolute_error", "squared_error"):
            raise ValueError(loss)
        self.loss, self.n_iter, self.lr = loss, n_iter, learning_rate
        self.max_leaf_nodes, self.min_leaf, self.max_bins = max_leaf_nodes, min_samples_leaf, max_bins

    # ---- binning
    def _make_bins(self, X: np.ndarray) -> None:
        qs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        self.n_bins_ = max(len(e) for e in self.edges_) + 1

    def _bin(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.searchsorted(self.edges_[j], X[:, j], side="left")
                                for j in range(X.shape[1])]).astype(np.uint8)

    # ---- one tree
    def _hist(self, Xb, g, idx, offsets):
        flat = (Xb[idx].astype(np.int32) + offsets).ravel()
        size = len(offsets) * self.n_bins_
        s = np.bincount(flat, weights=np.repeat(g[idx], len(offsets)), minlength=size)
        c = np.bincount(flat, minlength=size)
        return s.reshape(len(offsets), -1), c.reshape(len(offsets), -1)

    def _best_split(self, s, c):
        S, N = s[0].sum(), c[0].sum()
        cs, cc = np.cumsum(s, axis=1)[:, :-1], np.cumsum(c, axis=1)[:, :-1]
        nl, nr = cc, N - cc
        ok = (nl >= self.min_leaf) & (nr >= self.min_leaf)
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = cs ** 2 / nl + (S - cs) ** 2 / nr - S ** 2 / N
        gain = np.where(ok, gain, -np.inf)
        f, b = np.unravel_index(np.argmax(gain), gain.shape)
        return float(gain[f, b]), int(f), int(b)

    def _grow_tree(self, Xb, g, resid, offsets):
        n = len(g)
        feat, thr, left, right, val = [-1], [0], [-1], [-1], [0.0]
        root_idx = np.arange(n)
        s, c = self._hist(Xb, g, root_idx, offsets)
        leaves = {0: dict(idx=root_idx, hist=(s, c), split=self._best_split(s, c))}
        while len(leaves) < self.max_leaf_nodes:
            nid = max(leaves, key=lambda k: leaves[k]["split"][0])
            leaf = leaves[nid]; gain, f, b = leaf["split"]
            if not np.isfinite(gain) or gain <= 1e-12:
                break
            del leaves[nid]
            idx = leaf["idx"]; go_left = Xb[idx, f] <= b
            li, ri = idx[go_left], idx[~go_left]
            # histogram subtraction: build the smaller child, derive the other
            small_is_left = len(li) <= len(ri)
            sh = self._hist(Xb, g, li if small_is_left else ri, offsets)
            bh = (leaf["hist"][0] - sh[0], leaf["hist"][1] - sh[1])
            lh, rh = (sh, bh) if small_is_left else (bh, sh)
            lid, rid = len(feat), len(feat) + 1
            feat += [-1, -1]; thr += [0, 0]; left += [-1, -1]; right += [-1, -1]; val += [0.0, 0.0]
            feat[nid], thr[nid], left[nid], right[nid] = f, b, lid, rid
            leaves[lid] = dict(idx=li, hist=lh, split=self._best_split(*lh))
            leaves[rid] = dict(idx=ri, hist=rh, split=self._best_split(*rh))
        update = np.zeros(n)
        for nid, leaf in leaves.items():
            r = resid[leaf["idx"]]
            v = (np.median(r) if self.loss == "absolute_error" else r.mean()) * self.lr
            val[nid] = v; update[leaf["idx"]] = v
        tree = tuple(np.array(a) for a in (feat, thr, left, right, val))
        return tree, update

    # ---- public API
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self._make_bins(X); Xb = self._bin(X)
        offsets = (np.arange(X.shape[1]) * self.n_bins_).astype(np.int32)
        self.init_ = float(np.median(y) if self.loss == "absolute_error" else y.mean())
        F = np.full(len(y), self.init_); self.trees_ = []
        for _ in range(self.n_iter):
            resid = y - F
            g = np.sign(resid) if self.loss == "absolute_error" else resid
            tree, upd = self._grow_tree(Xb, g, resid, offsets)
            self.trees_.append(tree); F += upd
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = self._bin(np.asarray(X, dtype=float)); n = len(Xb); out = np.full(n, self.init_)
        rows = np.arange(n)
        for feat, thr, left, right, val in self.trees_:
            node = np.zeros(n, dtype=np.int64)
            while True:
                f = feat[node]; active = f >= 0
                if not active.any():
                    break
                a = rows[active]
                go_left = Xb[a, f[active]] <= thr[node[active]]
                node[a] = np.where(go_left, left[node[active]], right[node[active]])
            out += val[node]
        return out


class RidgeRegression:
    """Closed-form ridge (intercept not penalised)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        self.xm_, self.ym_ = X.mean(0), y.mean()
        Xc = X - self.xm_
        self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(X.shape[1]), Xc.T @ (y - self.ym_))
        return self

    def predict(self, X):
        return (np.asarray(X, dtype=float) - self.xm_) @ self.coef_ + self.ym_


def fit(X: pd.DataFrame, y: pd.Series, loss: str = "absolute_error") -> GradientBoosting:
    """Train boosting on log(rate): the target is right-skewed and errors become percentages."""
    return GradientBoosting(loss=loss).fit(X[FEATURES].to_numpy(float), np.log(np.asarray(y, float)))


def predict(model: GradientBoosting, X: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(X[FEATURES].to_numpy(float)))


def linear_xy(d: pd.DataFrame) -> pd.DataFrame:
    """Simple, stable features for the linear component: distance and equipment."""
    X = pd.DataFrame({"log_dist": np.log(d["distance"]), "dist": d["distance"]})
    for e in (0, 1, 2):
        X[f"eq{e}"] = (d["equip"] == e).astype(int)
        X[f"eq{e}_dist"] = X[f"eq{e}"] * d["distance"]
    return X


class BlendModel:
    """Weighted average (in log space) of gradient boosting and a ridge regression.

    Boosting (robust absolute-error loss) captures non-linear / interaction effects; the ridge
    model is very stable and smooths boosting's noise. Both predict log(rate).
    """

    def __init__(self, w_boost: float = 0.8):
        self.w = w_boost

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.boost = fit(X, y)
        self.linear = RidgeRegression(1.0).fit(linear_xy(X), np.log(np.asarray(y, float)))
        return self

    def boost_log(self, X): return self.boost.predict(X[FEATURES].to_numpy(float))
    def linear_log(self, X): return self.linear.predict(linear_xy(X))

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return self.w * self.boost_log(X) + (1 - self.w) * self.linear_log(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(X))


def metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE_%": float((np.abs(err) / np.abs(y_true)).mean() * 100),
    }


# --------------------------------------------------------------- December inputs
def build_december_features(december: pd.DataFrame, train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """december_chart_inputs.csv only has the date. The model also needs coordinates,
    market_index and quote_signal, so:
      * coordinates are copied from other Lexington / Fort Wayne rows in the data
      * market_index / quote_signal = daily mean of the December rows in validation.csv
        (those inputs are known for December even though the rate is not)
    """
    D = december.copy()
    D["date"] = pd.to_datetime(D["date"])
    rows = pd.concat([train, valid])
    pk = rows[rows["pickup"] == "Lexington"].iloc[0]
    dl = rows[rows["delivery"] == "Fort Wayne"].iloc[0]
    D["pickup_lat"], D["pickup_lon"] = pk["pickup_lat"], pk["pickup_lon"]
    D["delivery_lat"], D["delivery_lon"] = dl["delivery_lat"], dl["delivery_lon"]
    daily = valid[valid["date"] >= "2025-12-01"].groupby("date")[["market_index", "quote_signal"]].mean()
    daily = daily.reindex(D["date"]).interpolate().bfill().ffill()
    D["market_index"] = daily["market_index"].to_numpy()
    D["quote_signal"] = daily["quote_signal"].to_numpy()
    return D
