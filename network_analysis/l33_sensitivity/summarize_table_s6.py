"""Rebuild the numerical data for Table S6 from the archived module-level CSV."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('modules_csv', type=Path)
    parser.add_argument('output_csv', type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.modules_csv.open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            if row['descriptors'] == 'full':
                groups[row['group']].append(row)
    if not groups:
        parser.error('No rows with descriptors=full')
    rows = []
    for group, members in sorted(groups.items()):
        first, = [row for row in members if row['mic_rank'] == '1']
        values = [float(row['rho']) for row in members if row['rho']]
        rows.append((group, len(members), len(values),
                     median(values) if values else '', first['rho'], first['n']))
    with args.output_csv.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('group', 'total_modules', 'evaluable_modules', 'median_rho',
                         'lowest_mic_module_rho', 'lowest_mic_module_n'))
        writer.writerows(rows)


if __name__ == '__main__':
    main()
