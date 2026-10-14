"""Controller-level uncertainty and independent-run optimizer inference."""
from itertools import combinations
import hashlib
import numpy as np
import pandas as pd
from scipy import stats


def stable_seed(label, base=20261004):
    return int.from_bytes(hashlib.sha256(f"{base}:{label}".encode()).digest()[:4],"little")


def holm_adjust(values):
    p = np.asarray(values,float); out = np.full(len(p),np.nan)
    valid = np.flatnonzero(np.isfinite(p)); order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank,index in enumerate(order):
        running = max(running,min(1.0,(len(order)-rank)*p[index]))
        out[index] = running
    return out


def median_ci(values, seed=0, resamples=2000):
    x = np.asarray(values,float); x = x[np.isfinite(x)]
    if not len(x): return np.nan,np.nan
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(x,size=(resamples,len(x)),replace=True),axis=1)
    return tuple(np.quantile(medians,[.025,.975]))


def distribution_summary(frame, group_columns, metrics, resamples=2000):
    rows = []
    for key,group in frame.groupby(group_columns,sort=False,dropna=False):
        key = key if isinstance(key,tuple) else (key,)
        group_id = dict(zip(group_columns,key))
        for metric in metrics:
            if metric not in group: continue
            raw = pd.to_numeric(group[metric],errors="coerce").to_numpy(float)
            # Infinities (e.g. TTC with no closing) are recorded separately,
            # never silently presented as finite samples or counted as failure.
            x = raw[np.isfinite(raw)]
            low,high = median_ci(x,stable_seed(str(key)+metric),resamples)
            rows.append({**group_id,"metric":metric,"n_total":len(raw),"n_finite":len(x),
                "n_nan":int(np.isnan(raw).sum()),"n_positive_infinite":int(np.isposinf(raw).sum()),
                "median":float(np.median(x)) if len(x) else np.nan,
                "q1":float(np.quantile(x,.25)) if len(x) else np.nan,
                "q3":float(np.quantile(x,.75)) if len(x) else np.nan,
                "p05":float(np.quantile(x,.05)) if len(x) else np.nan,
                "p95":float(np.quantile(x,.95)) if len(x) else np.nan,
                "min":float(np.min(x)) if len(x) else np.nan,"max":float(np.max(x)) if len(x) else np.nan,
                "median_ci_low":low,"median_ci_high":high,
                "failed_controllers":int(group.failed.sum()) if "failed" in group else 0,
                "bootstrap_unit":"controller","ci_method":"percentile bootstrap, 95%"})
    return pd.DataFrame(rows)


def independent_comparisons(frame, group_column, metrics, resamples=2000):
    """No pairing inferred from run number or seed label across optimizers."""
    omnibus, pairs = [],[]
    groups = list(frame[group_column].drop_duplicates())
    for metric in metrics:
        arrays = {g:pd.to_numeric(frame.loc[frame[group_column]==g,metric],errors="coerce").to_numpy(float) for g in groups}
        arrays = {g:x[np.isfinite(x)] for g,x in arrays.items()}
        nonempty = [x for x in arrays.values() if len(x)]
        if len(nonempty) >= 2:
            if np.ptp(np.concatenate(nonempty)) == 0: statistic,p = 0.0,1.0
            else: statistic,p = stats.kruskal(*nonempty)
        else: statistic,p = np.nan,np.nan
        omnibus.append({"metric":metric,"test":"Kruskal-Wallis independent runs","statistic":statistic,"p_value":p,
                        "groups":len(nonempty),"pairing":"none"})
        current = []
        for a,b in combinations(groups,2):
            x,y = arrays[a],arrays[b]
            if not len(x) or not len(y): continue
            u,pair_p = stats.mannwhitneyu(x,y,alternative="two-sided",method="asymptotic")
            rng = np.random.default_rng(stable_seed(f"{metric}:{a}:{b}"))
            differences = (np.median(rng.choice(x,(resamples,len(x))),axis=1)
                           -np.median(rng.choice(y,(resamples,len(y))),axis=1))
            lo,hi = np.quantile(differences,[.025,.975])
            current.append({"metric":metric,"group_a":a,"group_b":b,"n_a":len(x),"n_b":len(y),
                "test":"Mann-Whitney U independent runs","U":u,"p_raw":pair_p,
                "cliffs_delta_a_minus_b":2*u/(len(x)*len(y))-1,
                "median_a":np.median(x),"median_b":np.median(y),
                "median_difference_a_minus_b":np.median(x)-np.median(y),
                "difference_ci_low":lo,"difference_ci_high":hi,
                "omnibus_rejected":bool(p < .05)})
        adjusted = holm_adjust([r["p_raw"] for r in current])
        for row,ph in zip(current,adjusted):
            row["p_holm"] = ph; row["p_bonferroni"] = min(1.0,row["p_raw"]*len(current))
            row["significant_holm"] = bool(ph < .05)
        pairs.extend(current)
    return pd.DataFrame(omnibus),pd.DataFrame(pairs)


def spearman_permutation(x,y,seed=0,resamples=1999):
    x,y = np.asarray(x,float),np.asarray(y,float)
    keep = np.isfinite(x)&np.isfinite(y); x,y = x[keep],y[keep]
    if len(x)<4 or np.ptp(x)==0 or np.ptp(y)==0: return len(x),np.nan,np.nan
    rx,ry = stats.rankdata(x),stats.rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    scale = np.sqrt(np.dot(rx,rx)*np.dot(ry,ry)); rho = np.dot(rx,ry)/scale
    rng = np.random.default_rng(seed); count = 0
    for _ in range(resamples):
        trial = np.dot(rx,rng.permutation(ry))/scale
        count += abs(trial) >= abs(rho)-1e-14
    return len(x),float(rho),float((count+1)/(resamples+1))
