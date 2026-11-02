"""Recovery selection verifies actual bytes; tensors are never loaded."""
import copy
import hashlib
import json

import pytest

from experiments.paper1_review.audit_completed import recovery_selection, digest


def saved(path,value):
    path.write_text(json.dumps(value,sort_keys=True))


@pytest.fixture
def evidence(tmp_path):
    parent=tmp_path/'fit'/'attempt1';parent.mkdir(parents=True)
    target=parent.with_name('attempt2');target.mkdir()
    sources={'source.py':'0'*64};source_digest=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    saved(parent/'run.json',dict(status='running',source_file_sha256=sources,source_sha256=source_digest))
    saved(parent/'config.json',{'frozen':True});(parent/'latest_checkpoint.pt').write_bytes(b'opaque never loaded')
    (target/'config.json').write_bytes((parent/'config.json').read_bytes())
    entry=dict(index=3,parent=str(parent),output=str(target),checkpoint=str(parent/'latest_checkpoint.pt'),
               checkpoint_sha256=digest(parent/'latest_checkpoint.pt'),parent_run_sha256=digest(parent/'run.json'),
               config_sha256=digest(parent/'config.json'),completed_updates=10)
    saved(target/'resume_dependency.json',{**{k:entry[k] for k in ('checkpoint','checkpoint_sha256','parent_run_sha256','completed_updates')},'logical_fit_id':str(parent)})
    proof=dict(directory=str(parent),run_sha256=entry['parent_run_sha256'],all_artifacts_verified=True,
               artifact_sha256={n:digest(parent/n) for n in ('config.json','latest_checkpoint.pt')})
    return dict(campaign={'trains':[dict(output=str(parent),index=3)]},recovery={'recoveries':[entry]},
                parents={'runs':[proof]},source_hashes=sources,source_digest=source_digest)


def test_valid_recovery_selects_continuation_and_preserves_parent_attestation(evidence):
    selection,parents=recovery_selection(**evidence)
    e=evidence['recovery']['recoveries'][0]
    assert selection=={e['parent']:e['output']}
    assert parents==evidence['parents']['runs'] and parents[0]['all_artifacts_verified']


@pytest.mark.parametrize('damage',['unattested','missing_checkpoint','changed_bytes','wrong_dependency','wrong_config','outside_campaign','duplicate','non_sibling'])
def test_recovery_invalid_evidence_is_rejected(evidence,damage):
    from pathlib import Path
    e=evidence['recovery']['recoveries'][0];p=evidence['parents']['runs'][0]
    if damage=='unattested':p.pop('all_artifacts_verified')
    elif damage=='missing_checkpoint':p['artifact_sha256'].pop('latest_checkpoint.pt')
    elif damage=='changed_bytes':Path(e['checkpoint']).write_bytes(b'changed')
    elif damage=='wrong_dependency':
        path=Path(e['output'])/'resume_dependency.json';d=json.loads(path.read_text());d['completed_updates']=11;saved(path,d)
    elif damage=='wrong_config':saved(Path(e['output'])/'config.json',{'frozen':False})
    elif damage=='outside_campaign':e['index']=4
    elif damage=='duplicate':evidence['recovery']['recoveries'].append(copy.deepcopy(e))
    elif damage=='non_sibling':e['output']=str(Path(e['output']).with_name('attempt3'))
    with pytest.raises(ValueError):recovery_selection(**evidence)


@pytest.mark.parametrize('status,numerical',[('completed',False),('failed',True)])
def test_recovery_rejects_completed_or_numerically_failed_parent(evidence,status,numerical):
    from pathlib import Path
    e=evidence['recovery']['recoveries'][0];p=evidence['parents']['runs'][0];path=Path(e['parent'])/'run.json'
    r=json.loads(path.read_text());r.update(status=status,numerical_failure=numerical);saved(path,r)
    e['parent_run_sha256']=p['run_sha256']=digest(path)
    with pytest.raises(ValueError,match='numerically failed parent'):recovery_selection(**evidence)


def test_full_auditor_emits_selected_override_and_explicit_parent(tmp_path,evidence):
    from pathlib import Path
    from experiments.paper1_review.audit_completed import audit, SOURCE_NAMES
    release=tmp_path/'release';release.mkdir()
    sources={}
    for name in SOURCE_NAMES:
        path=release/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('source '+name);sources[name]=digest(path)
    source_digest=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    protocol={'frozen':True};pd=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    entry=evidence['recovery']['recoveries'][0];parent=Path(entry['parent']);target=Path(entry['output'])
    config=dict(protocol=protocol,protocol_sha256=pd)
    for directory in (parent,target):saved(directory/'config.json',config)
    saved(parent/'run.json',dict(status='running',source_file_sha256=sources,source_sha256=source_digest))
    entry.update(parent_run_sha256=digest(parent/'run.json'),config_sha256=digest(parent/'config.json'))
    proof=evidence['parents']['runs'][0];proof.update(run_sha256=entry['parent_run_sha256']);proof['artifact_sha256']['config.json']=entry['config_sha256']
    saved(target/'resume_dependency.json',{**{k:entry[k] for k in ('checkpoint','checkpoint_sha256','parent_run_sha256','completed_updates')},'logical_fit_id':str(parent)})
    saved(target/'run.json',dict(status='completed',config_sha256=digest(target/'config.json'),source_file_sha256=sources,source_sha256=source_digest,
                                artifact_sha256={n:digest(target/n) for n in ('config.json','resume_dependency.json')}))
    campaign=dict(protocol_sha256=pd,initializations=[],trains=[dict(index=3,output=str(parent))],predictions=[])
    cp=release/'campaign_manifest.json';saved(cp,campaign)
    transfer=tmp_path/'transfer.json';saved(transfer,dict(protocol_sha256=pd,sources={**sources,'campaign_manifest.json':digest(cp)}))
    rp=tmp_path/'recovery.json';saved(rp,{**evidence['recovery'],'campaign_sha256':digest(cp)})
    pp=tmp_path/'parents.json';saved(pp,evidence['parents'])
    output=tmp_path/'audit'
    receipt=audit(cp,digest(cp),release,transfer,digest(transfer),output,recovery_manifest=rp,recovery_manifest_sha256=digest(rp),parent_audit=pp,parent_audit_sha256=digest(pp))
    result=json.loads((output/'completion_audit.json').read_text());overrides=json.loads((output/'overrides.json').read_text())
    assert result['completed_run_count']==1 and len(result['runs'])==2
    assert result['explicitly_verified_interrupted_parents']==1 and all(r['all_artifacts_verified'] for r in result['runs'])
    assert result['statuses'][0]['logical_directory']==str(parent)
    assert overrides[str(parent)]['audit_run_sha256']==digest(target/'run.json')
    assert receipt['artifact_sha256']['overrides.json']==digest(output/'overrides.json')
