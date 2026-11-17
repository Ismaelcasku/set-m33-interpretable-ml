#!/usr/bin/env python3
"""Summarize the corrected E. coli out-of-family tetramer predictions."""
import argparse
import csv
import itertools
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

METHODS = ("mean", "rf_original", "rf_offset", "rf_affine", "ridge_tetramers")
COUNTS = {4: 12, 5: 13, 6: 3}
EPS = 1e-12


def ranks(values):
    """Average ranks, including ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[start]] == values[order[end]]:
            end += 1
        for j in order[start:end]:
            result[j] = (start + 1 + end) / 2.0
        start = end
    return result


def spearman(a, b):
    x, y = ranks(a), ranks(b)
    m = (len(x) + 1) / 2.0
    dx, dy = [v - m for v in x], [v - m for v in y]
    den = math.sqrt(sum(v*v for v in dx) * sum(v*v for v in dy))
    return sum(u*v for u, v in zip(dx, dy)) / den if den else float("nan")


def pair_stats(rows, method, margin):
    correct = wrong = predicted_ties = 0
    for a, b in itertools.combinations(rows, 2):
        observed = a["log10_MIC"] - b["log10_MIC"]
        if abs(observed) <= margin + EPS:
            continue
        pred = a[method] - b[method]
        if abs(pred) <= EPS:
            predicted_ties += 1
        elif pred * observed > 0:
            correct += 1
        else:
            wrong += 1
    n = correct + wrong + predicted_ties
    rate = 100 * (correct + 0.5 * predicted_ties) / n if n else float("nan")
    return n, correct, wrong, predicted_ties, rate


def mae(rows, method):
    return statistics.mean(abs(r[method] - r["log10_MIC"]) for r in rows)


def load(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["arm_charge"] = int(r["arm_charge"])
        for k in ("MIC_uM", "log10_MIC") + METHODS:
            r[k] = float(r[k])
            if not math.isfinite(r[k]):
                raise ValueError("Non-finite value: {}".format(k))
        if not math.isclose(math.log10(r["MIC_uM"]), r["log10_MIC"], abs_tol=1e-9):
            raise ValueError("MIC/log10(MIC) mismatch")
    if Counter(r["arm_charge"] for r in rows) != COUNTS:
        raise ValueError("Unexpected charge counts")
    if len({r["peptide_id"] for r in rows}) != 28 or len({r["family"] for r in rows}) != 6:
        raise ValueError("Expected 28 unique peptides in six families")
    if any(abs(r["rf_original"]) > 10 for r in rows):
        raise ValueError("RF predictions do not look like corrected log10 MIC")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    rows = load(parser.parse_args().predictions)
    print("RANKING DENTRO DE CADA CARGA (predicciones guardadas)")
    print("carga,metodo,n,spearman,pares_MIC_distinta,acierto_pct,pares_MIC_ratio_mayor_2,acierto_pct,correctos_2,incorrectos_2,empates_pred_2")
    for charge, n in sorted(COUNTS.items()):
        group = [r for r in rows if r["arm_charge"] == charge]
        y = [r["log10_MIC"] for r in group]
        for method in METHODS:
            rho = spearman(y, [r[method] for r in group])
            all_pairs = pair_stats(group, method, 0)
            clear = pair_stats(group, method, math.log10(2))
            print("{},{},{},{:.3f},{},{:.1f},{},{:.1f},{},{},{}".format(
                charge, method, n, rho, all_pairs[0], all_pairs[-1],
                clear[0], clear[-1], clear[1], clear[2], clear[3]))
    print("\nMAE por familia (log10 de MIC en uM)")
    print("familia,n," + ",".join(METHODS))
    families = defaultdict(list)
    for r in rows:
        families[r["family"]].append(r)
    values = []
    for family, group in sorted(families.items()):
        errors = [mae(group, method) for method in METHODS]
        values.append(errors)
        print("{},{},".format(family, len(group)) + ",".join("{:.3f}".format(v) for v in errors))
    macro = [statistics.mean(v) for v in zip(*values)]
    micro = [mae(rows, method) for method in METHODS]
    print("promedio_igual_por_familia,6," + ",".join("{:.3f}".format(v) for v in macro))
    print("promedio_igual_por_peptido,28," + ",".join("{:.3f}".format(v) for v in micro))


if __name__ == "__main__":
    main()
