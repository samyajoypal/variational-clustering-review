"""Read-only completion/hash audit and optional label-free retrieval bundle.

Standard library only; never imports/executes a model or opens evaluation labels.
Run separately from the frozen training release. New output directories only.
The transfer-manifest hash must come from independently retained submission
records. Incomplete live attempts are statuses, not completed evidence.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

SOURCE_NAMES = {
    "experiments/paper1_review/" + n for n in
    ("final_runner.py", "final_protocol.py", "final_diagnostics.py", "models.py",
     "site_parameterization.py", "data.py", "workload.py", "provenance.py")
} | {"experiments/paper1_structured_inference/" + n for n in ("structured.py", "gaussian_reference.py")}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def safe(root,name):
    p=(Path(root)/name).resolve();p.relative_to(Path(root).resolve());return p


def read(path):return json.loads(Path(path).read_text())


def anchor(path,expected):
    if digest(path)!=expected:raise ValueError('Independent anchor mismatch: '+str(path))
    return read(path)


def selections(campaign,phase,index):
    items=[]
    if phase in ('all','initialize'):
        for row in campaign['initializations']:
            if index is None or row['index']==index:items.append(dict(phase='initialize',index=row['index'],row=row,directory=row['output']))
    if phase in ('all','train'):
        for row in campaign['trains']:
            if index is None or row['index']==index:items.append(dict(phase='train',index=row['index'],row=row,directory=row['output']))
    if phase in ('all','predict'):
        for row in campaign['predictions']:
            if index is None or row['index']==index:
                for condition in row['conditions']:items.append(dict(phase='predict',index=row['index'],condition=condition,row=row,directory=row['outputs'][condition]))
    if not items:raise ValueError('Empty or out-of-range selection')
    if phase=='all' and index is not None:raise ValueError('Choose one phase when selecting an index')
    return items


def recovery_selection(campaign, recovery, parents, source_hashes, source_digest):
    """Recheck pinned interrupted evidence without executing checkpoint state.

    This audit certifies bytes and the declared direct continuation link. The
    evaluator separately checks the full continuation contract before labels.
    """
    rows={r['output']:r for r in campaign['trains']}
    entries=recovery.get('recoveries',[])
    parents_by_path={r['directory']:r for r in parents.get('runs',[])}
    if len(parents_by_path)!=len(parents.get('runs',[])):raise ValueError('Duplicate parent evidence')
    if not entries or len({r['parent'] for r in entries})!=len(entries):raise ValueError('Empty or duplicate recovery selection')
    if set(parents_by_path)!={r['parent'] for r in entries}:raise ValueError('Parent evidence selection mismatch')
    selected={}
    for entry in entries:
        parent=Path(entry['parent']);target=Path(entry['output'])
        if str(parent) not in rows or rows[str(parent)]['index']!=entry['index']:
            raise ValueError('Recovery is outside the registered campaign')
        if parent.name!='attempt1' or target!=parent.with_name('attempt2'):
            raise ValueError('Recovery must select the registered sibling attempt2')
        proof=parents_by_path[str(parent)]
        if proof.get('all_artifacts_verified') is not True or not proof.get('artifact_sha256'):
            raise ValueError('Interrupted parent lacks explicit artifact attestation')
        if proof['run_sha256']!=entry['parent_run_sha256']:raise ValueError('Parent anchor mismatch')
        if entry['checkpoint']!=str(parent/'latest_checkpoint.pt') or proof['artifact_sha256'].get('latest_checkpoint.pt')!=entry['checkpoint_sha256']:
            raise ValueError('Recovery checkpoint missing from parent audit')
        if proof['artifact_sha256'].get('config.json')!=entry['config_sha256']:
            raise ValueError('Recovery configuration differs from parent audit')
        run=anchor(parent/'run.json',proof['run_sha256'])
        if run.get('status')=='completed' or run.get('numerical_failure'):
            raise ValueError('Completed or numerically failed parent cannot be recovered')
        if run.get('source_file_sha256')!=source_hashes or run.get('source_sha256')!=source_digest:
            raise ValueError('Interrupted parent source changed')
        for name,expected in proof['artifact_sha256'].items():
            if name.startswith('evaluation/'):raise ValueError('Unexpected evaluation artifact')
            if digest(safe(parent,name))!=expected:raise ValueError('Interrupted parent artifact changed: '+name)
        if digest(parent/'run.json')!=proof['run_sha256']:raise ValueError('Parent changed during audit')
        dep=read(target/'resume_dependency.json')
        for key in ('checkpoint','checkpoint_sha256','parent_run_sha256','completed_updates'):
            if dep.get(key)!=entry[key]:raise ValueError('Recovery dependency mismatch: '+key)
        if dep.get('logical_fit_id')!=str(parent):raise ValueError('Recovery logical fit mismatch')
        if digest(target/'config.json')!=entry['config_sha256']:raise ValueError('Recovery config changed')
        selected[str(parent)]=str(target)
    return selected,list(parents_by_path.values())


def audit(campaign_path,campaign_sha,release,transfer_path,transfer_sha,output,phase='all',index=None,bundle=False,
          recovery_manifest=None,recovery_manifest_sha256=None,parent_audit=None,parent_audit_sha256=None):
    output,release=Path(output),Path(release)
    if output.exists():raise FileExistsError(output)
    transfer=anchor(transfer_path,transfer_sha);campaign=anchor(campaign_path,campaign_sha)
    if transfer['sources'].get('campaign_manifest.json')!=campaign_sha or transfer['protocol_sha256']!=campaign['protocol_sha256']:
        raise ValueError('Campaign/transfer identity mismatch')
    for name,expected in transfer['sources'].items():
        if digest(safe(release,name))!=expected:raise ValueError('Frozen release changed: '+name)
    source_hashes={name:transfer['sources'][name] for name in sorted(SOURCE_NAMES)}
    source_digest=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    records,statuses,files={},[],{}
    selected,parent_records={},[]
    recovery_args=(recovery_manifest,recovery_manifest_sha256,parent_audit,parent_audit_sha256)
    if any(v is not None for v in recovery_args):
        if not all(v is not None for v in recovery_args):raise ValueError('All four recovery/parent anchors are required')
        if phase!='all' or index is not None or bundle:raise ValueError('Recovery audit requires all phases without bundling')
        recovery=anchor(recovery_manifest,recovery_manifest_sha256);parents=anchor(parent_audit,parent_audit_sha256)
        if recovery.get('campaign_sha256')!=campaign_sha:raise ValueError('Recovery campaign anchor mismatch')
        selected,parent_records=recovery_selection(campaign,recovery,parents,source_hashes,source_digest)
    protocol=None
    for item in selections(campaign,phase,index):
        if item['directory'] in selected:
            item=dict(item,logical_directory=item['directory'],directory=selected[item['directory']])
        directory=Path(item['directory']);run_path=directory/'run.json'
        status={k:v for k,v in item.items() if k!='row'}
        if not run_path.exists():statuses.append(dict(status,status='missing'));continue
        before=digest(run_path);run=read(run_path);statuses.append(dict(status,status=run.get('status'),run_sha256=before))
        if run.get('status')!='completed':continue
        if run.get('source_file_sha256')!=source_hashes or run.get('source_sha256')!=source_digest:raise ValueError('Run source differs from transferred release')
        config=read(directory/'config.json')
        if digest(directory/'config.json')!=run['config_sha256']:raise ValueError('Configuration changed')
        if config.get('protocol_sha256')!=campaign['protocol_sha256']:raise ValueError('Protocol changed')
        if 'protocol' in config:
            p=config['protocol'];pd=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if pd!=campaign['protocol_sha256'] or (protocol is not None and p!=protocol):raise ValueError('Full protocol mismatch')
            protocol=p
        hashes=run.get('artifact_sha256',{})
        if not hashes:raise ValueError('Completed run has no artifact inventory')
        for name,expected in hashes.items():
            path=safe(directory,name)
            if name.startswith('evaluation/'):raise ValueError('Learner artifact unexpectedly contains evaluation data')
            if digest(path)!=expected:raise ValueError('Artifact changed: '+str(path))
        if digest(run_path)!=before:raise ValueError('Run record changed during audit')
        records[str(directory)]=dict(directory=str(directory),run_sha256=before,artifact_sha256=hashes,
                                     all_artifacts_verified=True,status='completed',phase=item['phase'],index=item['index'])
        if bundle:
            relative=directory.relative_to(campaign['output_root'])
            files['matched_final_v1/'+str(relative/'run.json')]=(run_path,before)
            for name,expected in hashes.items():
                if name in {'latest_checkpoint.pt','previous_checkpoint.pt'} or (name.startswith('checkpoints/') and name.endswith('.pt')):continue
                files['matched_final_v1/'+str(relative/name)]=(safe(directory,name),expected)
            # Only declared input arrays are bundled. Evaluation-label/origin
            # arrays are never opened or copied by this compatibility auditor.
            row=item['row'];manifest_path=Path(row['manifest']);meta=read(manifest_path)
            expected=campaign['inputs_sha256'][meta['dataset']]
            if digest(manifest_path)!=expected:raise ValueError('Prepared dataset manifest changed')
            prefix='data_inductive_v1/'+meta['dataset']
            files[prefix+'/manifest.json']=(manifest_path,expected)
            for split in ('train','validation','test'):
                name=meta['splits'][split]['inputs'];path=safe(manifest_path.parent,name)
                h=meta['artifact_sha256'][name]
                if digest(path)!=h:raise ValueError('Prepared input changed')
                files[prefix+'/'+name]=(path,h)
            if row.get('training_view'):
                vp=Path(row['training_view']);vm=read(vp);prefix='training_views_v1/'+str(vp.parent.relative_to(Path(row['training_view']).parents[3]))
                files[prefix+'/manifest.json']=(vp,digest(vp))
                for split in ('train','validation'):
                    name=vm['splits'][split]['inputs'];path=safe(vp.parent,name);h=vm['artifact_sha256'][name]
                    if digest(path)!=h:raise ValueError('Training-view input changed')
                    files[prefix+'/'+name]=(path,h)
    if protocol is None:
        # The source protocol is not executed: use the independently transferred
        # freeze record, which contains the exact protocol JSON.
        freeze=read(release/'evidence/protocol_freeze.json')
        candidates=[freeze.get('protocol'),freeze]
        for p in candidates:
            if isinstance(p,dict) and hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':')).encode()).hexdigest()==campaign['protocol_sha256']:
                protocol=p;break
    if protocol is None:raise ValueError('Full frozen protocol unavailable; audit at least one completed initializer/fit')
    output.mkdir(parents=True,exist_ok=False)
    evidence=dict(schema_version='paper1-matched-completion-audit-v1',checked_utc=datetime.now(timezone.utc).isoformat(),
                  release=str(release),campaign_sha256=campaign_sha,transfer_manifest_sha256=transfer_sha,
                  auditor_sha256=digest(Path(__file__)),selection=dict(phase=phase,index=index),
                  all_recorded_completed_artifacts_verified=True,runs=list(records.values())+parent_records,statuses=statuses,
                  completed_run_count=len(records),verified_artifact_count=sum(len(r['artifact_sha256']) for r in records.values()),
                  labels_loaded=False,scope='Completed immutable selected runs only; live noncompleted attempts are status observations, not independently verified artifacts')
    if selected:
        evidence.update(recovery_manifest_sha256=recovery_manifest_sha256,parent_audit_sha256=parent_audit_sha256,
                        explicitly_verified_interrupted_parents=len(parent_records),
                        scope='Selected completed runs and explicitly reverified interrupted parents; byte integrity only, evaluator must also validate full lineage')
        if not all(target in records for target in selected.values()):raise ValueError('Selected continuation did not complete')
        write(output/'overrides.json',{original:dict(directory=target,reason='Audited scheduler preemption continuation',
              audit_run_sha256=records[target]['run_sha256']) for original,target in selected.items()})
    write(output/'completion_audit.json',evidence)
    write(output/'release_hashes.json',dict(protocol=protocol,protocol_sha256=campaign['protocol_sha256'],source_file_sha256=source_hashes,
             release=str(release),campaign_sha256=campaign_sha,transfer_manifest_sha256=transfer_sha,
             source_scope='Exact ten snapshots recorded by final_runner start_run; broader transfer set independently checked above'))
    write(output/'retrieval_manifest.json',dict(labels_loaded=False,files={k:dict(source=str(p),sha256=h,bytes=p.stat().st_size) for k,(p,h) in sorted(files.items())}))
    if bundle:
        with tarfile.open(output/'label_free_bundle.tar.gz','w:gz') as tar:
            for name,(path,expected) in sorted(files.items()):
                if digest(path)!=expected:raise ValueError('Artifact changed before bundling')
                tar.add(path,arcname=name,recursive=False)
    receipt=dict(schema_version='paper1-matched-audit-receipt-v1',labels_loaded=False,
                 output=str(output),artifact_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    write(output/'receipt.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['campaign','release','transfer-manifest','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--campaign-sha256',required=True);p.add_argument('--transfer-manifest-sha256',required=True)
    p.add_argument('--phase',choices=['all','initialize','train','predict'],default='all');p.add_argument('--index',type=int)
    p.add_argument('--recovery-manifest',type=Path);p.add_argument('--recovery-manifest-sha256')
    p.add_argument('--parent-audit',type=Path);p.add_argument('--parent-audit-sha256')
    p.add_argument('--bundle',action='store_true');a=p.parse_args()
    print(json.dumps(audit(a.campaign,a.campaign_sha256,a.release,a.transfer_manifest,a.transfer_manifest_sha256,a.output,a.phase,a.index,a.bundle,
                          a.recovery_manifest,a.recovery_manifest_sha256,a.parent_audit,a.parent_audit_sha256),sort_keys=True))


if __name__=='__main__':main()
