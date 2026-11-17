#!/usr/bin/env python3
import argparse
import csv
import functools
import itertools
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SPECIES = dict(E_coli="ecoli", P_aeruginosa="paeruginosa", K_pneumoniae="pneumoniae")
EXPECTED = dict(E_coli=37, P_aeruginosa=30, K_pneumoniae=22)
lines = [x.split() for x in (DATA / "BLOSUM62.txt").read_text().splitlines()]
AA = set(lines[0])
B = {(r[0], c): int(v) for r in lines[1:] for c, v in zip(lines[0], r[1:])}
assert len(B) == 400 and all(B[x,y] == B[y,x] for x,y in B)

def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with open(path, "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def mic(r, sp):
    if not r[sp].strip():
        return None
    v = float(r[sp])
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"Invalid MIC: {r['peptide_id']}, {sp}: {v}")
    return v

def peptides(path):
    rs = read(path)
    assert len(rs) == len({r["peptide_id"] for r in rs}) == 40
    for r in rs:
        assert r["sequence"] and set(r["sequence"]) <= AA
    assert {s: sum(mic(r,s) is not None for r in rs) for s in SPECIES} == EXPECTED
    return rs

@functools.lru_cache(maxsize=100000)
def align(a, b):
    # Needleman–Wunsch, BLOSUM62; linear gap cost 5, including terminal gaps.
    # On score ties prefer more identities, then more paired residues.
    prev = [(-5*j,0,0) for j in range(len(b)+1)]
    for i,x in enumerate(a,1):
        cur = [(-5*i,0,0)]
        for j,y in enumerate(b,1):
            d, dm, dp = prev[j-1]
            u, um, up = prev[j]
            l, lm, lp = cur[j-1]
            cur.append(max((d+B[x,y],dm+(x==y),dp+1),(u-5,um,up),(l-5,lm,lp)))
        prev = cur
    score, match, paired = prev[-1]
    norm = score / math.sqrt(sum(B[x,x] for x in a)*sum(B[y,y] for y in b))
    return norm, match/paired if paired else 0.0, paired/len(a), paired/len(b)

def families(rs):
    known = {r["peptide_id"]:r for r in read(DATA/"family_map.csv")}
    parent = list(range(len(rs)))
    def root(i):
        while i != parent[i]:
            i = parent[i]
        return i
    for i,j in itertools.combinations(range(len(rs)),2):
        x,y = rs[i],rs[j]
        kx,ky = known[x["peptide_id"]],known[y["peptide_id"]]
        assert kx["sequence"]==x["sequence"] and ky["sequence"]==y["sequence"]
        _, ident, c1, c2 = align(x["sequence"],y["sequence"])
        if kx["family"]==ky["family"] or (ident>=.8 and min(c1,c2)>=.8):
            parent[root(j)] = root(i)
    groups = {}
    for i,r in enumerate(rs):
        groups.setdefault(root(i),set()).add(known[r["peptide_id"]]["family"])
    return {r["peptide_id"]:"+".join(sorted(groups[root(i)])) for i,r in enumerate(rs)}

def audit(rs, out):
    groups=families(rs)
    ans=[]
    for sp,filename in SPECIES.items():
        train = read(DATA/(filename+"_mic.csv"))
        train = [r for r in train if (r["ID"], r["SEQUENCE"]) != ("ID","SEQUENCE")]
        train = [(r["ID"],r["SEQUENCE"].strip().upper()) for r in train]
        assert all(s and set(s)<=AA for _,s in train)
        for r in rs:
            seq=r["sequence"]
            hits=[(align(seq,t), tid,t) for tid,t in train]
            (norm,ident,c1,c2),tid,t = max(hits)
            top_covered=max((a[1] for a,_,_ in hits if min(a[2:])>=.8), default=0)
            exact=any(t==seq for _,t in train)
            v=mic(r,sp)
            if exact and v is not None:
                raise ValueError(f"Exact overlap still eligible: {sp} {r['peptide_id']}")
            ans.append(dict(species=sp,peptide_id=r["peptide_id"],sequence=seq,
                family=groups[r["peptide_id"]],MIC_uM=v,
                exact=exact, nearest_id=tid,nearest_sequence=t,blosum_norm=norm,
                identity=ident,coverage_arm=c1,coverage_train=c2,
                max_identity_with_80pct_both=top_covered,
                **{f"close_{q}":top_covered>=q/100 for q in (70,80,90)}))
    write(out/"sequence_audit.csv",ans)
    summary={}
    for sp in SPECIES:
        cohort=[r for r in ans if r["species"]==sp and r["MIC_uM"] is not None]
        summary[sp]={"n":len(cohort),"families":len({r["family"] for r in cohort}),
            **{f"n_without_close_{q}":sum(not r[f"close_{q}"] for r in cohort) for q in (70,80,90)}}
    print(json.dumps(summary,indent=2))

CHARGE_COUNTS = {4:12, 5:13, 6:3}
def charge(r):
    return sum(c in "KRH" for c in r["sequence"])

def matched(rs):
    cohort = [r for r in rs if mic(r,"E_coli") is not None and charge(r) in CHARGE_COUNTS]
    if {q:sum(charge(r)==q for r in cohort) for q in CHARGE_COUNTS} != CHARGE_COUNTS:
        raise ValueError("Unexpected E. coli cohort by nominal arm charge")
    return cohort

def prepare(a, rs):
    hdr = Path(a.header).read_text(encoding="utf-8-sig")
    before, _, tail = re.split(r"(?im)^\s*(@data)\s*(?:\r?\n|$)",hdr,maxsplit=1)
    pattern = r"""(?im)^\s*@attribute\s+(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"|(\S+))\s+(numeric|real|integer)\s*(?:%[^\n]*)?$"""
    attrs=[next(v for v in m.groups()[:3] if v is not None) for m in re.finditer(pattern,before)]
    if len(attrs)!=len(re.findall(r"(?im)^\s*@attribute\b",before)) or len(attrs)<2 or len(set(attrs))!=len(attrs):
        raise ValueError("Expected a header with unique numeric attributes")
    ci = attrs.index(a.target)
    if a.pool:
        pool=read(a.pool)
        if not pool:
            raise ValueError(f"Empty training pool: {a.pool}")
        cols=list(pool[0])
        missing=set(attrs)-set(cols)
        extra=set(cols)-set(attrs)
        if missing or extra-{"ID"}:
            raise ValueError(f"Header/pool mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        vals=sorted(float(r[a.target]) for r in pool)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("Nonfinite training targets")
        report=dict(charge=a.charge,target=a.target,training_rows=len(vals),
                    n_header_attributes=len(attrs),has_pool_ID="ID" in cols,
                    has_model_ID="ID" in attrs,order_matches=[k for k in cols if k in attrs]==attrs,
                    target_min=vals[0],target_median=vals[len(vals)//2],target_max=vals[-1])
        (a.out/"pool_check.json").write_text(json.dumps(report,indent=2)+"\n")
    if any(k.casefold()=="id" for k in attrs):
        raise ValueError("Model header contains ID. Verify original preprocessing; refusing to substitute peptide identifiers.")
    desc = {r["ID"]:r for r in read(DATA/"tetramers_qsar.csv")}
    selected=[r for r in rs if mic(r,a.species) is not None]
    if a.charge is not None:
        if a.species!="E_coli":
            raise ValueError("Charge matching is configured only for E. coli")
        selected=[r for r in matched(rs) if charge(r)==a.charge]
    rows=[]
    for r in selected:
        d=desc[r["peptide_id"]]
        vals=["?" if i==ci else d[name] for i,name in enumerate(attrs)]
        assert all(v=="?" if i==ci else math.isfinite(float(v)) for i,v in enumerate(vals))
        rows.append(",".join(vals))
    with (a.out/"test.arff").open("x") as f:
        f.write(before.rstrip()+"\n@data\n"+"\n".join(rows)+"\n")
    write(a.out/"row_map.csv",[dict(index=i+1,peptide_id=r["peptide_id"],sequence=r["sequence"],
        MIC_uM=mic(r,a.species)) for i,r in enumerate(selected)])
    print(ci+1)

def evaluate(a, rs):
    import numpy as np
    from scipy.stats import spearmanr
    audit_rows = {(r["species"],r["peptide_id"]):r for r in read(a.audit)}
    if a.matched_dir:
        subset=matched(rs)
        by_id={}
        for q,scale in zip(CHARGE_COUNTS,a.scales):
            batch=a.matched_dir/f"charge_{q}"
            cohort=[r for r in subset if charge(r)==q]
            mapping=read(batch/"row_map.csv")
            records=read(batch/"weka_predictions.csv")
            if len(records)!=len(mapping) or len(mapping)!=len(cohort):
                raise ValueError(f"Unexpected prediction count for +{q}")
            for j,(m,record,r) in enumerate(zip(mapping,records,cohort),1):
                i=r["peptide_id"]
                if (int(m["index"])!=j or int(record["inst#"])!=j or m["peptide_id"]!=i
                    or m["sequence"]!=r["sequence"] or i in by_id):
                    raise ValueError(f"Invalid prediction mapping for +{q}: {i}")
                v=float(record["predicted"])
                if not math.isfinite(v) or (scale=="raw_uM" and v<=0):
                    raise ValueError(f"Invalid prediction for {i}: {v}")
                by_id[i]=(v,math.log10(v) if scale=="raw_uM" else v,scale)
        p=np.array([by_id[r["peptide_id"]][1] for r in subset])
        write(a.out/"merged_rf_predictions.csv",[
            dict(peptide_id=i,arm_charge=charge(r),rf_raw=by_id[i][0],
                 rf_scale=by_id[i][2],rf_log10_MIC_uM=by_id[i][1])
            for r in subset for i in [r["peptide_id"]]])
    else:
        p,subset=read_single(a,rs)
    return evaluate_folds(a,subset,p,audit_rows)

def read_single(a, rs):
    import numpy as np
    raw = Path(a.predictions).read_text().splitlines()
    start = next(i for i,x in enumerate(raw) if x.strip().lower().startswith("inst#,"))
    records=list(csv.DictReader(x.strip() for x in raw[start:] if x.strip()))
    mapping=read(a.row_map)
    subset=[r for r in rs if mic(r,a.species) is not None]
    assert [r["peptide_id"] for r in mapping]==[r["peptide_id"] for r in subset]
    assert len(records)==len(mapping)
    assert [int(r["inst#"]) for r in records]==list(range(1,len(mapping)+1))
    p=np.array([float(r["predicted"]) for r in records])
    assert np.all(np.isfinite(p))
    if a.scale=="raw_uM":
        assert np.all(p>0), "RF raw MIC must be positive"
        p=np.log10(p)
    return p,subset

def evaluate_folds(a,subset,p,audit_rows):
    import numpy as np
    from scipy.stats import spearmanr
    y=np.log10([mic(r,a.species) for r in subset])
    ids=[r["peptide_id"] for r in subset]
    info=[audit_rows[a.species,i] for i in ids]
    assert all(t["sequence"]==r["sequence"] and float(t["MIC_uM"])==mic(r,a.species)
               for t,r in zip(info,subset))
    if any(r["exact"]!="False" or r["close_80"] not in ("True","False") for r in info):
        raise ValueError("Unexpected exact overlap or invalid audit flag in evaluation cohort")
    g=np.array([r["family"] for r in info])
    if a.matched_dir:
        summary=dict(n=len(subset),families=len(set(g)),n_without_close_80=sum(r["close_80"]=="False" for r in info))
        summary["by_charge"]={q:dict(n=sum(charge(r)==q for r in subset),
            n_without_close_80=sum(charge(r)==q and t["close_80"]=="False" for r,t in zip(subset,info)))
            for q in CHARGE_COUNTS}
        (a.out/"matched_cohort.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(json.dumps(summary,indent=2))
    if len(set(g))<3:
        raise ValueError("Fewer than three independent families; no pilot fit")
    d=read(DATA/"tetramers_qsar.csv")
    names=[k for k in d[0] if k!="ID"]
    d={r["ID"]:r for r in d}
    X=np.array([[float(d[i][k]) for k in names] for i in ids])
    assert np.isfinite(X).all()
    pred={m:np.full(len(y),np.nan) for m in ["mean","rf_original","rf_offset","rf_affine","ridge_tetramers"]}
    folds=[]
    for fam in sorted(set(g)):
        test=g==fam
        tr=~test
        base=y[tr].mean()
        pred["mean"][test]=base
        pred["rf_original"][test]=p[test]
        offset=np.mean(y[tr]-p[tr])
        pred["rf_offset"][test]=p[test]+offset
        if np.ptp(p[tr])<1e-10:
            slope,intercept=1.0,offset
        else:
            intercept,slope=np.linalg.lstsq(np.c_[np.ones(tr.sum()),p[tr]],y[tr],rcond=None)[0]
        pred["rf_affine"][test]=intercept+slope*p[test]
        mu=X[tr].mean(axis=0)
        sd=X[tr].std(axis=0)
        keep=sd>1e-12
        if keep.any():
            z=(X[:,keep]-mu[keep])/sd[keep]/math.sqrt(keep.sum())
            ztr=z[tr]
            coef=np.linalg.solve(ztr@ztr.T+np.eye(tr.sum()),y[tr]-base)
            pred["ridge_tetramers"][test]=base+(z[test]@ztr.T)@coef
        else:
            pred["ridge_tetramers"][test]=base
        folds.append(dict(held_out_family=fam,n_train=tr.sum(),n_test=test.sum(),
                          offset_log10=offset,affine_intercept=intercept,affine_slope=slope,
                          n_ridge_features=keep.sum()))
    write(a.out/"held_out_fits.csv",folds)
    write(a.out/"out_of_fold_predictions.csv",[
        dict(peptide_id=i,arm_charge=charge(subset[j]),family=g[j],MIC_uM=mic(subset[j],a.species),
             log10_MIC=y[j],close_80=info[j]["close_80"],**{m:v[j] for m,v in pred.items()})
        for j,i in enumerate(ids)])
    metrics=[]
    for label,mask in [("all",np.ones(len(y),dtype=bool))]+[
        (f"without_close_{q}",np.array([r[f"close_{q}"]=="False" for r in info])) for q in (70,80,90)]:
        for method, vals in pred.items():
            u,v=y[mask],vals[mask]
            err=v-u
            rho=spearmanr(u,v).statistic if len(u)>=3 and np.ptp(u)>0 and np.ptp(v)>0 else math.nan
            metrics.append(dict(subset=label,method=method,n=len(u),families=len(set(g[mask])),
                MAE_log10=np.abs(err).mean() if len(u) else "",RMSE_log10=np.sqrt(np.mean(err**2)) if len(u) else "",
                bias_log10=err.mean() if len(u) else "",spearman=rho if math.isfinite(rho) else ""))
    write(a.out/"metrics.csv",metrics)

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=True)
    for cmd in ("audit","prepare","evaluate"):
        s=sub.add_parser(cmd)
        s.add_argument("--mic",type=Path,required=True)
        s.add_argument("--out",type=Path,required=True)
        if cmd!="audit":
            s.add_argument("--species",choices=SPECIES,required=True)
        if cmd=="prepare":
            s.add_argument("--header",required=True)
            s.add_argument("--target",required=True)
            s.add_argument("--charge",type=int,choices=CHARGE_COUNTS)
            s.add_argument("--pool",type=Path)
        if cmd=="evaluate":
            for opt in ("audit","row-map","predictions"):
                s.add_argument("--"+opt,type=Path,required=opt=="audit")
            s.add_argument("--scale",choices=["raw_uM","log10_uM"])
            s.add_argument("--matched-dir",type=Path)
            s.add_argument("--scales",nargs=3,choices=["raw_uM","log10_uM"],metavar=("PLUS4","PLUS5","PLUS6"))
    a=p.parse_args()
    rs=peptides(a.mic)
    a.out.mkdir(parents=True,exist_ok=True)
    if a.cmd=="audit": audit(rs,a.out)
    elif a.cmd=="prepare": prepare(a,rs)
    else:
        if a.matched_dir:
            if a.species!="E_coli" or not a.scales or any([a.row_map,a.predictions,a.scale]):
                p.error("Matched evaluation requires E_coli, --matched-dir, and three --scales")
        elif not (a.row_map and a.predictions and a.scale) or a.scales:
            p.error("Single-model evaluation requires --row-map, --predictions, and --scale")
        evaluate(a,rs)

if __name__=="__main__": main()
