"""
Metricas (definiciones identicas a las usadas en E1/E5).

  metrics      <- experimentos_fase_C/E1_comparar.py L22-28
  paired       <- experimentos_fase_C/E1_comparar.py L31-40 (bootstrap de diferencias por sujeto + Wilcoxon)
  ic95_runs    <- E1_comparar.py L56 (1.96 * sd / sqrt(n) sobre las 33 corridas)
  metrics_e5   <- experimentos_fase_C/E5_comparar.py L22-26 (metr)
  boot_subj    <- experimentos_fase_C/E5_comparar.py L29-32
  mae_por_sujeto <- E5_comparar.py L39 (MAE medio por sujeto)
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def metrics(y, p):
    e = p - y; a = np.abs(e)
    return {"n": len(y), "MAE": a.mean(), "RMSE": np.sqrt((e ** 2).mean()), "MedAE": np.median(a),
            "pearson_r": pearsonr(y, p)[0], "spearman_rho": spearmanr(y, p).correlation, "bias": e.mean(),
            "abs_p25": np.percentile(a, 25), "abs_p50": np.percentile(a, 50), "abs_p75": np.percentile(a, 75),
            "abs_p90": np.percentile(a, 90), "pct_<=3": 100 * (a <= 3).mean(), "pct_<=5": 100 * (a <= 5).mean(),
            "pct_<=10": 100 * (a <= 10).mean()}


def paired(d, label, rng):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    bs = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    try:
        p = wilcoxon(d).pvalue
    except ValueError:
        p = np.nan
    return {"comparacion": label, "n_sujetos": len(d), "dif_media": d.mean(), "dif_mediana": np.median(d),
            "ic95_bootstrap_sujetos": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "wilcoxon_p": p, "pct_sujetos_mejoran(dif<0)": 100 * (d < 0).mean()}


def ic95_runs(values):
    v = np.asarray(values, float)
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def metrics_e5(y, p):
    e = p - y; a = np.abs(e)
    return {"MAE": a.mean(), "RMSE": np.sqrt((e ** 2).mean()), "MedAE": np.median(a), "bias": e.mean(),
            "pearson": pearsonr(y, p)[0] if np.std(p) > 0 else np.nan, "spearman": spearmanr(y, p).correlation if np.std(p) > 0 else np.nan,
            "pct_<=3": 100 * (a <= 3).mean(), "pct_<=5": 100 * (a <= 5).mean(), "pct_<=10": 100 * (a <= 10).mean()}


def boot_subj(d, rng, n=5000):
    v = d.values
    bs = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mae_por_sujeto(df, col, y="bpm_watch", subject="participante"):
    return (df[col] - df[y]).abs().groupby(df[subject]).mean()
