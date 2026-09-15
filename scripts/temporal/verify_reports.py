"""Audit indépendant des CSV/rapports par comparaisons par paires, sans modèle ni fit."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median


def auc(labels, scores):
    positive = [s for y,s in zip(labels,scores) if y == 1]
    negative = [s for y,s in zip(labels,scores) if y == 0]
    return sum(float(p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def verify(root):
    registration = json.loads((root / 'registration.json').read_text())
    names = ['c1_median', 'c1_ewma', 'orderless', 'lstm']
    group_sets = [{r['condition_group_id'] for r in registration['records'] if r['partition'] == fold}
                  for fold in ('train','validation','confirmation')]
    assert all(not a & b for i,a in enumerate(group_sets) for b in group_sets[i+1:])
    baseline = None
    for fold in ('validation','confirmation'):
        records = {r['recording_id']: r for r in registration['records'] if r['partition'] == fold}
        report = json.loads((root / fold / 'report.json').read_text())
        with (root / fold / 'predictions.csv').open() as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == ['recording_id',*names]
            rows = list(reader)
        assert len(rows) == len(records) and {r['recording_id'] for r in rows} == set(records)
        actual = {}
        for name in names:
            grouped = defaultdict(list)
            labels, scores = [], []
            for row in rows:
                record = records[row['recording_id']]
                label, score = int(record['label']=='leak'), float(row[name])
                assert math.isfinite(score) and 0 <= score <= 1
                labels.append(label); scores.append(score)
                grouped[record['condition_group_id']].append((label,score))
            assert abs(auc(labels,scores) - report['metrics'][name]['clip_level']['roc_auc']) <= 5.1e-7
            ys, ps = [], []
            for group in grouped.values():
                assert len(group) == 2 and len({y for y,p in group}) == 1
                ys.append(group[0][0]); ps.append(median(p for y,p in group))
            actual[name] = auc(ys,ps)
            assert abs(actual[name] - report['group_auc'][name]) < 1e-12
        if fold == 'validation':
            baseline = max(names[:-1], key=lambda n:actual[n])
        assert report['selected_baseline'] == baseline
        assert abs(actual['lstm']-actual[baseline]-report['lstm_auc_gain']) < 1e-12
        assert report['event_metrics'] is None and not report['official_zenodo_val_test_used']
        print(f'{fold}: mapping, scores, AUC clip/groupe et référence conformes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence',type=Path)
    verify(parser.parse_args().evidence)
