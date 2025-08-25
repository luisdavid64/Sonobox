import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats

def make_feature_matrix(df, drop_cols=("file","segment","cluster")):
    feat_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    # drop cols that are all NaN
    X = X.dropna(axis=1, how="all")
    feat_cols = list(X.columns)
    # rowwise impute remaining NaNs with column medians
    X = X.fillna(X.median(numeric_only=True))
    return X.values, feat_cols

def rank_features(df, labels, method="anova", scale=True, min_groups=2):
    """
    df: DataFrame with numeric feature columns and (optionally) 'file','segment'
    labels: array-like cluster or pseudo labels (-1 will be ignored)
    method: 'anova' or 'kruskal'
    scale: z-score features before ranking
    returns: DataFrame [feature, score, p, score_norm]
    """
    X, feat_cols = make_feature_matrix(df)
    labels = np.asarray(labels)
    valid = labels != -1  # ignore noise if using HDBSCAN
    X = X[valid]; y = labels[valid]

    if scale:
        X = StandardScaler().fit_transform(X)

    results = []
    uniq = [g for g in np.unique(y) if g != -1]
    if len(uniq) < min_groups:
        raise ValueError("Not enough groups to rank features.")

    for j, name in enumerate(feat_cols):
        groups = [X[y==g, j] for g in uniq]
        # some groups might be tiny—skip if any group has <2 samples for ANOVA
        if any(len(g) < 2 for g in groups) and method == "anova":
            continue
        try:
            if method == "anova":
                F, p = stats.f_oneway(*groups)
                score = F
            elif method == "kruskal":
                H, p = stats.kruskal(*groups, nan_policy="omit")
                score = H
            else:
                raise ValueError("method must be 'anova' or 'kruskal'")
            results.append((name, float(score), float(p)))
        except Exception:
            # robust to numerical issues
            continue

    out = pd.DataFrame(results, columns=["feature","score","p"]).sort_values("score", ascending=False)
    if not out.empty:
        out["score_norm"] = out["score"] / (out["score"].iloc[0] if out["score"].iloc[0] != 0 else 1.0)
    return out

def plot_feature_ranking(df_rank, top_k=20, score_col="score_norm", title=None):
    if df_rank.empty:
        print("No ranked features to plot.")
        return
    use_col = score_col if score_col in df_rank.columns else "score"
    top = df_rank.sort_values(use_col, ascending=True).tail(top_k)
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top[use_col])
    plt.xlabel("Normalized relevance" if use_col=="score_norm" else "Relevance score")
    plt.title(title or f"Top {len(top)} Features by Relevance")
    plt.tight_layout()
    plt.show()