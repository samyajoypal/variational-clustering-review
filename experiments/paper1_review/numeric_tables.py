"""Export all six presentation panels as full-precision, long-form numeric CSV."""
from __future__ import annotations
import csv
import json
from pathlib import Path

from .report_matched import PRIMARY, ALL_METRICS, FIELDS, validate_evaluation

IDENTITY=('dataset','method','method_a','method_b','contrast','scope','training_condition',
          'inference_condition','mode','condition')
COLUMNS=('panel','quantity',*IDENTITY,'metric','n','seeds','values_by_seed','mean','sample_sd',
         'median','minimum','maximum','interval_low','interval_high')

def export_tables(evaluation, output):
    lookup,readiness=validate_evaluation(evaluation)
    if readiness['partial']:raise ValueError('Complete five-seed panels required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    panels={name:[] for name in ['clean_main','clean_nine_metrics','conditional_nine_metrics',
                                'paired_all_metrics','paired_changes_main','paired_contrasts_ari']}
    def add(panel,quantity,row,metrics,context=None):
        for metric in metrics:
            score=row['scores'][metric];interval=score.get('descriptive_t_interval_95',['',''])
            value={'panel':panel,'quantity':quantity,**{k:row.get(k,'') for k in IDENTITY},
                'metric':metric,'n':row['n'],'seeds':json.dumps(row['seeds']),
                'values_by_seed':json.dumps(score['values_by_seed']),
                **{k:score[k] for k in ['mean','sample_sd','median','minimum','maximum']},
                'interval_low':interval[0],'interval_high':interval[1]}
            if context:value.update(context)
            panels[panel].append(value)
    for key,row in sorted(lookup['summaries'].items()):
        if key[2:]==('clean','clean'):
            add('clean_main','absolute',row,PRIMARY)
            add('clean_nine_metrics','absolute',row,ALL_METRICS)
        else:add('conditional_nine_metrics','absolute',row,ALL_METRICS)
    for name in FIELDS:
        if name=='summaries':continue
        for key,row in sorted(lookup[name].items()):
            add('paired_all_metrics',name,row,tuple(row['scores']))
            if name=='paired_condition_summaries':add('paired_changes_main','Delta',row,PRIMARY)
            if name=='condition_contrast_summaries':
                context={k:row[k] for k in ['mode','condition']}
                add('paired_contrasts_ari','D_clean',lookup['clean_contrast_summaries'][key[:5]],('ari',),context)
                add('paired_contrasts_ari','D_condition',row,('ari',))
                add('paired_contrasts_ari','Gamma',lookup['robustness_contrast_summaries'][key],('ari',))
    for name,rows in panels.items():
        with (output/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=COLUMNS);writer.writeheader();writer.writerows(rows)
    return {name:len(rows) for name,rows in panels.items()}
