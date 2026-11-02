"""Synthetic complete-roster fixture for integrity checks; submits no jobs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PARENT = Path(__file__).resolve().parents[1] / 'runs/campaign'
RELEASE = PARENT / 'pipeline_release_v10_matched_final'
BASE = PARENT / 'matched_final_v1'
VIEWS = PARENT / 'pipeline_release_v3_contamination/results/paper1_review_20261028/training_views_v1'
MANIFEST = 'campaign_manifest.json'
DATA_HASHES = {
    'mnist': '52d9d0f6ddaa65088ed881c9787a7c025e41ad3e999a772a233cec7412cdf008',
    'fashion_mnist': '5d514c069416d51af2623452cb31d1fb41a3b920103945b5065e7ccf59fdfcc1',
    'usps': 'a445d2787c61dc1ef9f835425dd1fc62fce5ec3c1daa5625e1ebb1f9eeed1cfa',
    'reuters10k': 'ce827811a448c208f746c9f940fd13e929f550df03e79bc25cdcc3d7d482edc7',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_manifest():
    from experiments.paper1_review.workload import plan
    from experiments.paper1_review.final_protocol import protocol_digest
    approved = plan()
    fits = [row for row in approved['real_data_fits'] if row['track'] != 'native_validation']
    groups, trains, predictions = {}, [], []
    corruptions = {}
    for row in approved['corruption_evaluations']:
        corruptions.setdefault(row['trained_run'], []).append(row['condition'])
    for row in fits:
        condition = row.get('condition', 'clean')
        key = f"{row['dataset']}/{condition}/seed{row['seed']}"
        manifest = str(PARENT / 'data_inductive_v1' / row['dataset'] / 'manifest.json')
        view = None if condition == 'clean' else str(VIEWS / key / 'manifest.json')
        if key not in groups:
            groups[key] = dict(index=len(groups), group=key, dataset=row['dataset'], condition=condition,
                               seed=row['seed'], manifest=manifest, training_view=view,
                               output=str(BASE / 'initializations' / key / 'attempt1'))
        initial = groups[key]
        train = dict(index=len(trains), run_id=row['run_id'], dataset=row['dataset'],
                     method=row['method'], seed=row['seed'], condition=condition,
                     manifest=manifest, training_view=view, initial=initial['output'],
                     initial_index=initial['index'], output=str(BASE / 'fits' / row['run_id'] / 'attempt1'))
        trains.append(train)
        conditions = ['clean', *corruptions.get(row['run_id'], [])]
        predictions.append(dict(index=len(predictions), run_id=row['run_id'], manifest=manifest,
                                fitted=train['output'], conditions=conditions,
                                outputs={c: str(BASE / 'predictions' / row['run_id'] / c / 'attempt1') for c in conditions}))
    result = dict(schema='paper1-air-matched-campaign-v1',
                  release=str(RELEASE), output_root=str(BASE), protocol_sha256=protocol_digest(),
                  counts=dict(initializations=40, clean_fits=200, contamination_fits=100,
                              training_fits=300, prediction_bundles=300, clean_test_predictions=300,
                              corruption_predictions=900),
                  inputs_sha256=DATA_HASHES, initializations=list(groups.values()),
                  trains=trains, predictions=predictions, labels_loaded=False)
    validate_manifest(result)
    return result


def validate_manifest(m):
    if m['release'] != str(RELEASE) or m['output_root'] != str(BASE):
        raise ValueError('Unexpected campaign identity')
    initial, trains, predictions = m['initializations'], m['trains'], m['predictions']
    if (len(initial), len(trains), len(predictions)) != (40, 300, 300):
        raise ValueError('Wrong stage counts')
    if sum(r['condition'] == 'clean' for r in trains) != 200:
        raise ValueError('Wrong clean/contaminated allocation')
    if sum(len(r['conditions']) - 1 for r in predictions) != 900:
        raise ValueError('Wrong corruption allocation')
    outputs = [r['output'] for r in initial + trains] + [p for r in predictions for p in r['outputs'].values()]
    if len(outputs) != len(set(outputs)) or any(not Path(p).is_relative_to(BASE) for p in outputs):
        raise ValueError('Duplicate or out-of-campaign outputs')
    for index, (train, pred) in enumerate(zip(trains, predictions)):
        init = initial[train['initial_index']]
        if train['index'] != index or pred['index'] != index or train['output'] != pred['fitted']:
            raise ValueError('Misaligned fit/prediction mapping')
        for field in ['manifest', 'training_view', 'seed', 'condition']:
            if init[field] != train[field]:
                raise ValueError('Initialization pairing changed')
        expected = 7 if train['condition'] == 'clean' and train['dataset'] != 'reuters10k' else 1
        if len(pred['conditions']) != expected or pred['conditions'][0] != 'clean':
            raise ValueError('Unexpected prediction scope')

