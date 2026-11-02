"""Verify distributed checksums and the complete portable scientific report."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def safe(relative):
    path=ROOT/relative
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError(f'Unsafe manifest path: {relative}')
    return path

def verify():
    manifest=json.loads((ROOT/'provenance/public_manifest.json').read_text())
    if manifest['schema_version']!='paper1-public-release-v1':raise ValueError('Unexpected public schema')
    for relative,expected in manifest['file_sha256'].items():
        if sha(safe(relative))!=expected:raise ValueError(f'Public checksum mismatch: {relative}')
    analysis_dir=ROOT/'results/paper1_review_20261031/matched_analysis_complete_20261031_v1'
    analysis=json.loads((analysis_dir/'analysis_manifest.json').read_text())
    for relative,expected in analysis['artifact_sha256'].items():
        path=analysis_dir/relative
        if not path.resolve().is_relative_to(analysis_dir):raise ValueError('Unsafe analysis artifact')
        if sha(path)!=expected:raise ValueError(f'Analysis checksum mismatch: {relative}')
    evaluation=json.loads((analysis_dir/'evaluation.json').read_text())
    if evaluation['provenance']!=analysis['provenance']:raise ValueError('Analysis provenance differs')
    from experiments.paper1_review.report_matched import validate_evaluation
    _,readiness=validate_evaluation(evaluation)
    if readiness['partial']:raise ValueError('Incomplete scientific analysis')
    report_dir=ROOT/'results/paper1_review_20261031/matched_report_complete_v1'
    report=json.loads((report_dir/'report_manifest.json').read_text())
    if report['partial'] or not report['score_panels_emitted']:raise ValueError('Incomplete report')
    if report['provenance']['analysis_manifest_sha256']!=sha(analysis_dir/'analysis_manifest.json'):
        raise ValueError('Report points to a different analysis')
    for relative,expected in report['artifact_sha256'].items():
        path=report_dir/relative
        if not path.resolve().is_relative_to(report_dir):raise ValueError('Unsafe report artifact')
        if sha(path)!=expected:raise ValueError(f'Report checksum mismatch: {relative}')
    diagnostics=ROOT/'results/paper1_review_20261031/matched_diagnostics_complete_20261031_v1'
    receipt=json.loads((diagnostics/'receipt.json').read_text())
    if receipt['diagnostics_sha256']!=sha(diagnostics/'diagnostics.json'):raise ValueError('Diagnostic checksum mismatch')
    if receipt['provenance']['analysis_manifest_sha256']!=sha(analysis_dir/'analysis_manifest.json'):
        raise ValueError('Diagnostic analysis binding differs')
    return {'status':'passed','files':len(manifest['file_sha256']),
            'counts':readiness['verified'],'partial':False,'report_figures':len(report['figures'])}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
