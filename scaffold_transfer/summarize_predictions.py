"""Rebuild the numerical data of Table S8 and Table S7 from archived inputs."""
import argparse
import csv
import io
import itertools
import math
import tarfile
from collections import defaultdict
from pathlib import Path

SOURCES = (
    'ecoli_out_of_fold_predictions.csv',
    'paer_kpn_predictions_490536.csv',
    'results_kpn5_compare_490662/predictions.csv',
)
SPECIES = ('E_coli', 'P_aeruginosa', 'K_pneumoniae')
GROUPS = list(itertools.product(SPECIES, (4, 5, 6)))


def summarize(bundle, mic_path):
    with mic_path.open(newline='', encoding='utf-8-sig') as handle:
        mic = {r['peptide_id']: r for r in csv.DictReader(handle)}
    groups = defaultdict(dict)
    with tarfile.open(bundle) as archive:
        for source in SOURCES:
            with io.TextIOWrapper(archive.extractfile(source), encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    species = row.get('species', 'E_coli')
                    charge = int(row.get('charge', row.get('arm_charge')))
                    peptide = row['peptide_id']
                    observed = float(row.get('MIC_tetramer_uM', row.get('MIC_uM')))
                    predicted = float(row.get('rf_output', row.get('rf_original')))
                    seq = mic[peptide]['sequence']
                    key = species, charge
                    if key == ('K_pneumoniae', 5) and row.get('model') != 'batch5kleb60':
                        raise ValueError('Wrong K. pneumoniae +5 model')
                    if not (observed > 0 and math.isfinite(observed) and math.isfinite(predicted)
                            and observed == float(mic[peptide][species])
                            and charge == sum(seq.count(aa) for aa in 'KRH')
                            and peptide not in groups[key]):
                        raise ValueError(f'Inconsistent input: {key}, {peptide}')
                    groups[key][peptide] = (predicted, observed, seq)
    if set(groups) != set(GROUPS):
        raise ValueError('Expected exactly nine species and charge groups')
    ranks, counts = [], []
    for species, charge in GROUPS:
        values = groups[species, charge]
        expected = {pid for pid, row in mic.items() if row[species] and
                    sum(row['sequence'].count(aa) for aa in 'KRH') == charge}
        if set(values) != expected:
            raise ValueError(f'Panel mismatch: {species}, +{charge}')
        assigned = {}
        for i, (pid, (prediction, observed, seq)) in enumerate(
                sorted(values.items(), key=lambda item: (item[1][0], item[0])), 1):
            ranks.append(dict(species=species, arm_charge=charge, peptide_id=pid,
                              linear_arm_sequence=seq,
                              rf_predicted_rank=assigned.setdefault(prediction, i),
                              experimental_tetramer_mic_uM=observed))
        eligible = concordant = ties = 0
        for (p1, m1, _), (p2, m2, _) in itertools.combinations(values.values(), 2):
            if max(m1, m2) <= 2 * min(m1, m2):
                continue
            eligible += 1
            concordant += (p1 - p2) * (m1 - m2) > 0
            ties += p1 == p2
        counts.append(dict(species=species, arm_charge=charge, n_peptides=len(values),
                           concordant_pairs=concordant, eligible_pairs=eligible,
                           tied_predictions=ties))
    return ranks, counts


def add_coverage(counts, ecoli_path, paer_kpn_path):
    """Add coverage from the two archived projections at the original thresholds."""
    covered = defaultdict(dict)
    for path, ecoli in ((ecoli_path, True), (paer_kpn_path, False)):
        with path.open(newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                if ecoli:
                    flag = row['historical_node'].lower()
                    if flag not in ('true', 'false'):
                        raise ValueError(f'Invalid historical_node flag: {flag}')
                    if flag == 'true':
                        continue
                    key = 'E_coli', int(row['charge'])
                else:
                    label, charge = row['group'].rsplit('_', 1)
                    key = {'paeruginosa': 'P_aeruginosa',
                           'kpneumoniae': 'K_pneumoniae'}[label], int(charge.removesuffix('plus'))
                pid, contacts = row['peptide_id'], int(row['n_contacts'])
                if pid in covered[key] or contacts < 0:
                    raise ValueError(f'Invalid coverage row: {key}, {pid}')
                covered[key][pid] = contacts > 0
    if set(covered) != set(GROUPS):
        raise ValueError('Expected coverage of nine species and charge groups')
    for row in counts:
        hits = covered[row['species'], row['arm_charge']]
        row.update(external_arms_with_neighbors=sum(hits.values()),
                   external_arms_evaluated=len(hits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prediction_bundle', 'mic_csv', 'output_dir'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--coverage', nargs=2, type=Path, metavar=('ECOLI_CSV', 'PAER_KPN_CSV'))
    args = parser.parse_args()
    ranks, counts = summarize(args.prediction_bundle, args.mic_csv)
    if args.coverage:
        add_coverage(counts, *args.coverage)
    s7_name = 'table_s7.csv' if args.coverage else 'table_s7_concordance.csv'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in [('table_s8.csv', ranks), (s7_name, counts)]:
        with (args.output_dir / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
