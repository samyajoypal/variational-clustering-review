"""Verify public-export anchors and regenerate all matched report tables/figures."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.paper1_review.report_matched import generate_report
from experiments.paper1_review.numeric_tables import export_tables

ANALYSIS=ROOT/'results/paper1_review_20261031/matched_analysis_complete_20261031_v1/analysis_manifest.json'
SHA256='379db7e40fed218d52dcd4d759babbab5f4215541c54eb2b1efd3617cf820c44'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New output directory')
    args=parser.parse_args()
    value=generate_report(ANALYSIS,args.output,analysis_manifest_sha256=SHA256)
    tables=export_tables(json.loads((ANALYSIS.parent/'evaluation.json').read_text()),args.output/'tables')
    for path in (args.output/'tables').glob('*.tex'):path.unlink()
    note=args.output/'README.txt'
    note.write_text(note.read_text().replace('LaTeX fragments require booktabs, tabularx, array and longtable. They are not a standalone document.',
        'Tables are full-precision long-form numeric CSV, with seed values and descriptive intervals.'))
    value['provenance']['numeric_csv_export_source_sha256']=hashlib.sha256((ROOT/'experiments/paper1_review/numeric_tables.py').read_bytes()).hexdigest()
    value['numeric_csv_rows']=tables
    value['artifact_sha256']={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='report_manifest.json'}
    (args.output/'report_manifest.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':value['status'],'partial':value['partial'],
        'figures':len(value['figures']),'manifest_sha256':hashlib.sha256((args.output/'report_manifest.json').read_bytes()).hexdigest()},indent=2))
