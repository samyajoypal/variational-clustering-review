"""Synthetic final-style artifacts; no model execution or benchmark labels."""
import copy
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from experiments.paper1_review import evaluate_matched as em
from experiments.paper1_review.data import _ids_hash
from experiments.paper1_review.final_protocol import PROTOCOL, protocol_digest
from experiments.paper1_review.provenance import sha256, write_json
from campaign_fixture import make_manifest, PARENT, BASE


class Fixture:
    def __init__(self, root):
        self.root = root
        self.campaign = make_manifest()
        self.sources = {n: hashlib.sha256(('fixture '+n).encode()).hexdigest() for n in em.SOURCE_FILES}
        self.release = dict(protocol=PROTOCOL, protocol_sha256=protocol_digest(), source_file_sha256=self.sources)
        self.audit = {'runs': []}
        self.mapper = em.Paths([(str(PARENT), str(root/'release')), (str(BASE), str(root/'outputs'))])
        self.maps = [f'{PARENT}={root / "release"}', f'{BASE}={root / "outputs"}']
        self.data = {}
        for name in em.DATASETS:
            path = root/'release'/'data_inductive_v1'/name/'manifest.json'; path.parent.mkdir(parents=True)
            meta = dict(schema_version='paper1-review-dataset-v1',status='completed',dataset=name,n_features=4,n_clusters=2,splits={},artifact_sha256={})
            self.data[name] = {}
            for split,count in [('train',6),('validation',4),('test',6)]:
                ids=np.array([f'{name}:{split}:{i}' for i in range(count)])
                x=np.random.default_rng(count).uniform(0,1,(count,4)).astype(np.float32); labels=np.arange(count)%2
                np.savez(path.parent/f'{split}.npz',x=x,ids=ids)
                np.savez(path.parent/f'{split}_labels.npz',labels=labels,ids=ids)
                meta['splits'][split]=dict(inputs=f'{split}.npz',labels=f'{split}_labels.npz',n_samples=count,input_shape=list(x.shape),ids_sha256=_ids_hash(ids))
                for f in [f'{split}.npz',f'{split}_labels.npz']:meta['artifact_sha256'][f]=sha256(path.parent/f)
                self.data[name][split]=dict(x=x,ids=ids,labels=labels)
            write_json(path,meta);self.campaign['inputs_sha256'][name]=sha256(path)
        self.refresh_anchors()

    def refresh_anchors(self):
        for name,data in [('campaign',self.campaign),('release',self.release),('audit',self.audit)]: write_json(self.root/f'{name}.json',data)

    def row(self,method='gaussian_q_gaussian_prior',seed=101,condition='clean'):
        return next(r for r in self.campaign['trains'] if r['dataset']=='mnist' and r['method']==method and r['seed']==seed and r['condition']==condition)

    def seal(self,remote,config=None,extra=None):
        p=self.mapper(remote)
        if config is not None:
            p.mkdir(parents=True,exist_ok=True);write_json(p/'config.json',config)
            for name in self.sources:
                target=p/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('fixture '+name)
            run=dict(schema_version='paper1-review-run-v2',protocol_version=em.PROTOCOL_VERSION,fresh_run=True,
                     legacy_outputs_used=False,stage='final_benchmark',status='completed',labels_loaded=False,
                     test_partition_loaded=False,source_file_sha256=self.sources,source_sha256=em._digest(self.sources))
        else:run=json.loads((p/'run.json').read_text())
        if extra:run.update(extra)
        run['config_sha256']=sha256(p/'config.json')
        run['artifact_sha256']={str(f.relative_to(p)):sha256(f) for f in sorted(p.rglob('*')) if f.is_file() and f.name!='run.json'}
        write_json(p/'run.json',run)
        evidence=dict(directory=remote,run_sha256=sha256(p/'run.json'),artifact_sha256=run['artifact_sha256'],all_artifacts_verified=True)
        self.audit['runs']=[r for r in self.audit['runs'] if r['directory']!=remote]+[evidence]
        self.refresh_anchors();return run

    def config(self,row,phase):
        d=self.data[row['dataset']];n=len(d['train']['ids']);epochs=100 if phase=='initialize' else 200
        ids=d['train']['ids'];order=sorted(range(n),key=lambda i:hashlib.sha256(str(ids[i]).encode()).digest())
        config=dict(protocol_version=em.PROTOCOL_VERSION,protocol=PROTOCOL,protocol_sha256=protocol_digest(),
            phase='fresh_final_shared_initialization' if phase=='initialize' else 'fixed_final_matched_fit',
            seed=row['seed'],dataset=row['dataset'],condition=row['condition'],clean_train_count=n,training_count=n,
            validation_count=len(d['validation']['ids']),clean_manifest_sha256=self.campaign['inputs_sha256'][row['dataset']],
            training_view_sha256=self.campaign['inputs_sha256'][row['dataset']],training_view_manifest=row['manifest'],
            training_ids_sha256=_ids_hash(ids),validation_ids_sha256=_ids_hash(d['validation']['ids']),common_gradient_ids=ids[order[:32]].tolist(),
            validation_only=False,labels_loaded=False,test_partition_loaded=False,native_reproduction=False,
            initialization_from_development=False,optimizer_updates=epochs)
        if row['training_view']:
            vp=self.mapper(row['training_view']);meta=json.loads(vp.read_text())
            config.update(training_view_sha256=sha256(vp),training_view_manifest=row['training_view'],
                          training_count=meta['splits']['train']['n_samples'],validation_count=meta['splits']['validation']['n_samples'],
                          training_ids_sha256=meta['splits']['train']['ids_sha256'],validation_ids_sha256=meta['splits']['validation']['ids_sha256'])
        return config

    def ids(self,row,split):
        if row['training_view']:
            vp=self.mapper(row['training_view']);meta=json.loads(vp.read_text())
            with np.load(vp.parent/meta['splits'][split]['inputs']) as a:return a['ids']
        return self.data[row['dataset']][split]['ids']

    def prepare_view(self,row):
        from experiments.paper1_review.contamination import prepare
        prepare(self.mapper(row['manifest']),self.mapper(row['training_view']).parent,condition=row['condition'],seed=row['seed'])

    def optimization(self,p,row,epochs,*,training=False):
        p.mkdir(parents=True,exist_ok=True);ids=self.ids(row,'train');n=len(ids)
        lengths=np.full(epochs,n,dtype=np.int32);np.save(p/'batch_lengths.npy',lengths)
        np.tile(np.arange(n,dtype='<i4'),epochs).tofile(p/'sample_order.int32')
        np.savez(p/'sample_exposure.npz',sample_ids=ids,count=np.full(n,epochs,dtype=np.int64))
        curves=[dict(clean_equivalent_epoch=i,updates=i,loss=1/(i+1),clipped_updates=0) for i in range(1,epochs+1)]
        write_json(p/('training_curves.json' if training else 'pretrain_curves.json'),curves)

    def initialize(self,row):
        remote=row['initial'];p=self.mapper(remote)
        if (p/'run.json').exists():return
        config=self.config(row,'initialize');self.optimization(p,row,100)
        (p/'autoencoder.pt').write_bytes(b'never deserialized shared AE')
        np.savez(p/'mixture.npz',loc=np.zeros((2,10)),covariance_diag=np.ones((2,10)),weights=[.5,.5])
        self.seal(remote,config,dict(optimizer_updates=100,gmm={'converged':True},neural_seconds=2.))

    def predfile(self,p,split='test',permute=False,ids=None):
        d=self.data['mnist'][split];ids=d['ids'] if ids is None else ids;pred=np.arange(len(ids))%2
        if permute:pred=1-pred
        prob=np.eye(2)[pred]*.8+.1
        np.savez(p,sample_ids=ids,predictions=pred,responsibilities=prob,cluster_labels=np.arange(2))

    def fit(self,row):
        self.initialize(row);p=self.mapper(row['output']);self.optimization(p,row,200,training=True)
        initial=self.mapper(row['initial']);ir=json.loads((initial/'run.json').read_text())
        config=self.config(row,'train');config.update(method=row['method'],initial_run_sha256=sha256(initial/'run.json'),initial_tensor_sha256=ir['artifact_sha256']['autoencoder.pt'])
        (p/'model.pt').write_bytes(('never deserialized '+row['run_id']).encode())
        q=self.q(len(self.ids(row,'validation'))) if row['method']=='structured_q_student_prior' else None
        write_json(p/'numerical_checks.json',[dict(clean_equivalent_epoch=e,quadrature=q) for e in PROTOCOL['diagnostic_epochs']])
        components={'target_kl'} if row['method']=='dec_matched' else ({'target_kl','reconstruction'} if row['method']=='idec_matched' else {'reconstruction','prior_cross_entropy','negative_entropy'})
        write_json(p/'gradient_diagnostics.json',[dict(clean_equivalent_epoch=e,completed_updates=e,optimizer_update=False,components={k:dict(value=0.,preclip_norm=0.,group_norms={}) for k in components}) for e in PROTOCOL['diagnostic_epochs']])
        for split in ['train','validation']:self.predfile(p/f'{split}_predictions.npz',split,ids=self.ids(row,split))
        self.seal(row['output'],config,dict(optimizer_updates=200,numerical_gate_passed=True,fixed_endpoint=True,final_gmm={'converged':True} if row['method']=='vae_gmm' else None,training_seconds=3.))
        return row

    def q(self,n=6):
        s=PROTOCOL['structured'];return dict(passed=True,training_nodes_pass=True,prediction_nodes_pass=True,reference_refinement_pass=True,selected_positions=list(range(n)),
            rows=[dict(nodes=n,bounds=b,errors=dict(log_normalizer=0.,responsibilities=0.,kl_to_prior=0.)) for n,b in [(128,s['bounds']),(256,s['bounds']),(512,s['reference_bounds']),(1024,s['refined_reference_bounds'])]])

    def prediction(self,row,condition='clean',permute=False):
        pr=self.campaign['predictions'][row['index']];remote=pr['outputs'][condition];p=self.mapper(remote);p.mkdir(parents=True)
        fit=self.mapper(row['output']);run=json.loads((fit/'run.json').read_text());d=self.data['mnist']['test']
        verifier=em.Verifier(self.campaign,self.release,self.audit,self.mapper,{})
        config=dict(protocol_version=em.PROTOCOL_VERSION,phase='fixed_checkpoint_prediction',split='test',corruption=condition,
                    model_sha256=run['artifact_sha256']['model.pt'],fitted_run_sha256=sha256(fit/'run.json'),clean_manifest_sha256=self.campaign['inputs_sha256']['mnist'],
                    labels_loaded=False,seed=row['seed'],method=row['method'],protocol_sha256=protocol_digest(),
                    evaluated_input_sha256=verifier.input_hash(row,condition),sample_ids_sha256=_ids_hash(d['ids']))
        self.predfile(p/'predictions.npz',permute=permute)
        write_json(p/'numerical_checks.json',dict(quadrature=self.q() if row['method']=='structured_q_student_prior' else None))
        self.seal(remote,config,dict(checkpoint_sha256=config['model_sha256']))
        return remote

    def kwargs(self):
        self.refresh_anchors()
        return dict(campaign_sha256=sha256(self.root/'campaign.json'),release_hashes=self.root/'release.json',release_hashes_sha256=sha256(self.root/'release.json'),
                    completion_audit=self.root/'audit.json',completion_audit_sha256=sha256(self.root/'audit.json'),path_map=self.maps)

    def evaluate(self,name='analysis',**kwargs):
        return em.evaluate(self.root/'campaign.json',self.root/name,**self.kwargs(),**kwargs)


@pytest.fixture
def fixture(tmp_path):return Fixture(tmp_path)


def test_inventory_and_strict_incomplete_never_read_labels(fixture,monkeypatch):
    row=fixture.fit(fixture.row());fixture.prediction(row)
    monkeypatch.setattr(em,'load_evaluation_labels',lambda *a:pytest.fail('labels opened before completeness gate'))
    result=fixture.evaluate('inventory',inventory_only=True)
    assert result['verified']==dict(initializations=1,fits=1,predictions=1)
    assert result['labels_loaded'] is False and len(result['missing_or_failed'])==1537
    assert (fixture.root/'inventory/analysis_manifest.json').exists()
    assert result['predictions'][0]['logical_identity']['method']==row['method']
    with pytest.raises(ValueError,match='Incomplete campaign'):fixture.evaluate('strict')
    assert json.loads((fixture.root/'strict/FAILED.json').read_text())['labels_may_have_been_loaded'] is False


def test_partial_five_seed_aggregates_contrasts_and_frozen_map(fixture):
    for seed in em.SEEDS:
        for method in ['gaussian_q_gaussian_prior','gaussian_q_student_prior']:
            row=fixture.fit(fixture.row(method,seed));fixture.prediction(row)
            fixture.prediction(row,'dropout0.25',permute=method=='gaussian_q_student_prior')
    result=fixture.evaluate(allow_incomplete=True)
    groups=[r for r in result['summaries'] if r['dataset']=='mnist' and r['method']=='gaussian_q_student_prior']
    clean=next(r for r in groups if r['inference_condition']=='clean')
    assert clean['n']==5 and clean['scores']['ari']['mean']==1.
    assert next(r for r in groups if r['inference_condition']=='gaussian_sd0.1')['status']=='incomplete_seed_set_aggregate_withheld'
    change=next(r for r in result['paired_condition_rows'] if r['method']=='gaussian_q_student_prior')
    assert change['scores']['ari']==0 and change['scores']['frozen_acc']==-1.
    assert change['mapping_changes']['any_map_change']
    contrast=next(r for r in result['robustness_contrast_summaries'] if r['contrast']=='prior_family_under_gaussian_q' and r['condition']=='dropout0.25' and r['dataset']=='mnist')
    assert contrast['n']==5 and contrast['scores']['frozen_acc']['descriptive_t_interval_95']==[-1.,-1.]
    assert (fixture.root/'analysis/seed_metrics.csv').exists()
    with pytest.raises(FileExistsError):fixture.evaluate(allow_incomplete=True)


def test_missing_seed_withholds_pair_and_summary(fixture):
    for seed in em.SEEDS[:-1]:
        row=fixture.fit(fixture.row(seed=seed));fixture.prediction(row)
    result=fixture.evaluate(allow_incomplete=True)
    group=next(r for r in result['summaries'] if r['dataset']=='mnist' and r['method']=='gaussian_q_gaussian_prior' and r['inference_condition']=='clean')
    assert group['missing_seeds']==[105] and 'scores' not in group
    assert all('scores' not in r for r in result['clean_contrast_summaries'])


@pytest.mark.parametrize('damage',['lineage','short_budget','source','duplicate_ids','corruption_hash','quadrature','probabilities','input_bytes','audit_hash'])
def test_invalid_completed_endpoints_abort_before_any_labels(fixture,monkeypatch,damage):
    method='structured_q_student_prior' if damage=='quadrature' else 'gaussian_q_gaussian_prior'
    row=fixture.fit(fixture.row(method));remote=fixture.prediction(row,'dropout0.25');fit=fixture.mapper(row['output']);pred=fixture.mapper(remote)
    if damage=='lineage':
        c=json.loads((fit/'config.json').read_text());c['initial_tensor_sha256']='f'*64;write_json(fit/'config.json',c);fixture.seal(row['output'])
    elif damage=='short_budget':
        lengths=np.load(fit/'batch_lengths.npy');np.save(fit/'batch_lengths.npy',lengths[:-1]);fixture.seal(row['output'])
    elif damage=='source':
        r=json.loads((fit/'run.json').read_text());r['source_file_sha256']=dict(r['source_file_sha256']);key=next(iter(r['source_file_sha256']));r['source_file_sha256'][key]='f'*64
        r['source_sha256']=em._digest(r['source_file_sha256']);write_json(fit/'run.json',r);fixture.seal(row['output'])
    elif damage in ['duplicate_ids','probabilities']:
        with np.load(pred/'predictions.npz') as a:values={k:a[k] for k in a.files}
        if damage=='duplicate_ids':values['sample_ids'][1]=values['sample_ids'][0]
        else:values['responsibilities'][0]=[.9,.9]
        np.savez(pred/'predictions.npz',**values);fixture.seal(remote)
    elif damage=='corruption_hash':
        c=json.loads((pred/'config.json').read_text());c['evaluated_input_sha256']='f'*64;write_json(pred/'config.json',c);fixture.seal(remote)
    elif damage=='quadrature':
        checks=json.loads((fit/'numerical_checks.json').read_text());checks[-1]['quadrature']['rows'][0]['errors']['log_normalizer']=.5;write_json(fit/'numerical_checks.json',checks);fixture.seal(row['output'])
    elif damage=='input_bytes':
        path=fixture.mapper(row['manifest']).parent/'train.npz';path.write_bytes(b'changed dataset')
    else:
        run=json.loads((fit/'run.json').read_text());run['training_seconds']=0.;write_json(fit/'run.json',run)
    monkeypatch.setattr(em,'load_evaluation_labels',lambda *a:pytest.fail('invalid endpoint opened labels'))
    with pytest.raises((ValueError,FileNotFoundError)):
        fixture.evaluate(allow_incomplete=True)


def test_trusted_anchor_and_campaign_roster_cannot_be_replaced(fixture):
    args=fixture.kwargs();fixture.campaign['trains'][0]['seed']=999;fixture.refresh_anchors()
    with pytest.raises(ValueError,match='anchor hash'):em.evaluate(fixture.root/'campaign.json',fixture.root/'bad',**args)
    with pytest.raises(ValueError,match='roster'):fixture.evaluate()


def test_descriptive_interval_matches_independent_formula_and_no_significance():
    from scipy.stats import t
    values=np.array([-.2,.1,.0,.2,.4]);result=em.describe(values,interval=True)
    half=t.ppf(.975,4)*np.std(values,ddof=1)/np.sqrt(5)
    np.testing.assert_allclose(result['descriptive_t_interval_95'],[values.mean()-half,values.mean()+half])
    assert 'p_value' not in result
    with pytest.raises(ValueError):em.describe([1,2,3,4],interval=True)


def test_prediction_reordering_is_explicit_and_duplicate_ids_rejected(tmp_path):
    path=tmp_path/'p.npz';ids=np.array(['a','b','c']);prob=np.array([[.9,.1],[.1,.9],[.7,.3]])
    np.savez(path,sample_ids=ids[::-1],predictions=prob.argmax(1)[::-1],responsibilities=prob[::-1],cluster_labels=[0,1])
    p=em._prediction(path,ids,2);np.testing.assert_array_equal(p['probabilities'],prob)
    with pytest.raises(ValueError,match='IDs differ'):em._prediction(path,np.array(['a','b','d']),2)


def test_frozen_accepted_roundoff_normalizes_without_changing_any_partition_score(tmp_path):
    path=tmp_path/'roundoff.npz';ids=np.array(['a','b','c','d'])
    raw=np.array([[1.0000009536743164,1e-8],[1e-8,1.0000009536743164],[.6000005,.4000005],[.2,.7999995]],dtype=np.float32)
    pred=raw.argmax(1);truth=np.array([0,1,1,0])
    np.savez(path,sample_ids=ids,predictions=pred,responsibilities=raw,cluster_labels=[0,1])
    original=sha256(path);p=em._prediction(path,ids,2);e=p['probability_normalization']
    assert sha256(path)==original and e['original_artifact_modified'] is False
    assert e['raw_entries_above_one']==2 and e['raw_maximum']>1
    assert e['raw_native_max_row_sum_error']<1e-5 and e['raw_float64_max_row_sum_error']<1e-5
    assert e['max_absolute_adjustment']<2e-6 and e['changed_rows']==4
    assert e['normalized_argmax_unchanged'] and e['normalization_dtype']=='float64'
    np.testing.assert_array_equal(p['predictions'],pred)
    assert (p['probabilities']<=1).all()
    np.testing.assert_allclose(p['probabilities'].sum(1),1,atol=1e-15,rtol=0)
    bare=em.clustering_metrics(truth,pred)['scores']
    prepared=em.clustering_metrics(truth,pred,probabilities=p['probabilities'],cluster_labels=[0,1])['scores']
    assert len(bare)==9 and prepared==bare


@pytest.mark.parametrize('damage',['large_mass_error','negative','wrong_argmax'])
def test_roundoff_normalization_never_repairs_invalid_arrays(tmp_path,damage):
    p=tmp_path/'invalid.npz';raw=np.array([[1.+1e-6,0.],[.1,.9]],dtype=np.float32);pred=raw.argmax(1)
    if damage=='large_mass_error':raw[0,0]=1.+2e-5
    elif damage=='negative':raw[0,1]=-1e-8
    else:pred[0]=1
    np.savez(p,sample_ids=['a','b'],predictions=pred,responsibilities=raw,cluster_labels=[0,1])
    with pytest.raises(ValueError,match='Invalid probability'):em._prediction(p,np.array(['a','b']),2)


def test_probability_normalization_evidence_reaches_fit_prediction_and_origin(fixture):
    row=fixture.row(method='student_q_student_prior',condition='gaussian_training_sd0.3_ratio0.1')
    fixture.prepare_view(row);fixture.fit(row);fixture.prediction(row)
    result=fixture.evaluate(allow_incomplete=True)
    inventory=json.loads((fixture.root/'analysis/inventory.json').read_text())
    fit=inventory['fits'][0];prediction=inventory['predictions'][0]
    assert set(fit['probability_normalization'])=={'train_predictions.npz','validation_predictions.npz'}
    assert set(prediction['probability_normalization'])=={'predictions.npz'}
    origin=json.loads((fixture.root/'analysis/training_origin_allocation.json').read_text())[0]
    assert origin['probability_normalization']['raw_artifact_sha256']==origin['training_prediction_sha256']
    assert all(v['normalized_argmax_unchanged'] for v in fit['probability_normalization'].values())


def test_resume_requires_trusted_receipt_and_identical_committed_prefix(fixture):
    row=fixture.fit(fixture.row());parent_remote=row['output'];parent=fixture.mapper(parent_remote)
    (parent/'latest_checkpoint.pt').write_bytes(b'opaque audited checkpoint')
    write_json(parent/'latest_checkpoint.json',dict(sha256=sha256(parent/'latest_checkpoint.pt'),phase='epoch_boundary',resumable=True,completed_updates=100,source_sha256=em._digest(fixture.sources)))
    fixture.seal(parent_remote,extra={'status':'running'})
    child_remote=str(Path(parent_remote).with_name('attempt2'));child=fixture.mapper(child_remote)
    import shutil
    shutil.copytree(parent,child)
    write_json(child/'resume_dependency.json',dict(checkpoint=str(Path(parent_remote)/'latest_checkpoint.pt'),checkpoint_sha256=sha256(parent/'latest_checkpoint.pt'),
        parent_run_sha256=sha256(parent/'run.json'),completed_updates=100,logical_fit_id=parent_remote))
    fixture.seal(child_remote,extra={'status':'completed','logical_fit_id':parent_remote})
    override=fixture.root/'overrides.json';write_json(override,{parent_remote:dict(directory=child_remote,reason='audited scheduler preemption',audit_run_sha256=sha256(child/'run.json'))})
    result=fixture.evaluate('resumed',inventory_only=True,overrides=override,overrides_sha256=sha256(override))
    assert result['verified']['fits']==1 and len(result['fits'][0]['resume_lineage'])==1
    assert result['fits'][0]['cost']['total_physical_cost_available'] is False
    order=np.fromfile(child/'sample_order.int32',dtype='<i4');order[:6]=order[:6][::-1];order.tofile(child/'sample_order.int32')
    fixture.seal(child_remote)
    replacements=json.loads(override.read_text());replacements[parent_remote]['audit_run_sha256']=sha256(child/'run.json');write_json(override,replacements)
    with pytest.raises(ValueError,match='prefix changed'):
        fixture.evaluate('forged',inventory_only=True,overrides=override,overrides_sha256=sha256(override))


def test_better_condition_score_does_not_mean_better_gamma(fixture):
    a,b='gaussian_q_student_prior','gaussian_q_gaussian_prior'
    for seed in em.SEEDS:
        for method in [a,b]:
            row=fixture.fit(fixture.row(method,seed))
            for condition in ['clean','dropout0.25']:
                remote=fixture.prediction(row,condition);p=fixture.mapper(remote)/'predictions.npz'
                with np.load(p) as saved:values={k:saved[k] for k in saved.files}
                flips=[] if method==a and condition=='clean' else ([0] if method==a else [0,1])
                values['predictions'][flips]=1-values['predictions'][flips]
                values['responsibilities']=np.eye(2)[values['predictions']]*.8+.1
                np.savez(p,**values);fixture.seal(remote)
    result=fixture.evaluate(allow_incomplete=True)
    gamma=next(r for r in result['robustness_contrast_summaries'] if r['dataset']=='mnist' and r['contrast']=='prior_family_under_gaussian_q' and r['condition']=='dropout0.25')
    difference=next(r for r in result['condition_contrast_summaries'] if r['dataset']=='mnist' and r['contrast']=='prior_family_under_gaussian_q' and r['condition']=='dropout0.25')
    assert difference['scores']['hungarian_acc']['mean']==pytest.approx(1/6)
    assert gamma['scores']['hungarian_acc']['mean']==pytest.approx(-1/6)
    assert difference['scores']['hungarian_acc']['descriptive_t_interval_95']==pytest.approx([1/6,1/6])


def test_training_contamination_pairing_origin_and_no_frozen_map(fixture):
    method='structured_q_gaussian_prior';condition='gaussian_training_sd0.3_ratio0.1'
    for seed in em.SEEDS:
        clean=fixture.fit(fixture.row(method,seed));fixture.prediction(clean)
        contaminated=fixture.row(method,seed,condition);fixture.prepare_view(contaminated)
        fixture.fit(contaminated);fixture.prediction(contaminated)
    result=fixture.evaluate(allow_incomplete=True)
    rows=[r for r in result['paired_condition_rows'] if r['mode']=='retrained']
    assert len(rows)==5 and all(r['clean_map_frozen'] is None for r in rows)
    allocations=json.loads((fixture.root/'analysis/training_origin_allocation.json').read_text())
    assert len(allocations)==5 and all(r['status']=='completed' for r in allocations)
    assert all(sum(c['clean'] for c in r['clusters'])==6 for r in allocations)
    assert all(sum(c['contaminants'] for c in r['clusters'])==r['contaminant_count'] for r in allocations)
    assert all('semantic_accuracy' not in r for r in allocations)


def test_missing_origin_is_pending_not_silently_scored(fixture):
    row=fixture.row('vae_gmm',101,'gaussian_training_sd0.3_ratio0.1');fixture.prepare_view(row);fixture.fit(row);fixture.prediction(row)
    origin=fixture.mapper(row['training_view']).parent/'evaluation/train_origin.npz';origin.unlink()
    fixture.evaluate(allow_incomplete=True)
    allocations=json.loads((fixture.root/'analysis/training_origin_allocation.json').read_text())
    assert allocations[0]['status']=='pending_origin_artifact'


def test_global_audit_attestation_or_explicit_per_run_required(fixture):
    row=fixture.fit(fixture.row());fixture.audit['runs'][0].pop('all_artifacts_verified');fixture.refresh_anchors()
    with pytest.raises(ValueError,match='attest'):fixture.evaluate(inventory_only=True)


def test_numerical_failure_remains_in_partial_ledger(fixture,monkeypatch):
    row=fixture.fit(fixture.row());fixture.seal(row['output'],extra=dict(status='failed',numerical_failure=True,error='attained quadrature failed'))
    monkeypatch.setattr(em,'load_evaluation_labels',lambda *a:pytest.fail('no completed predictions'))
    result=fixture.evaluate(inventory_only=True)
    failed=[r for r in result['missing_or_failed'] if r['status']=='failed']
    assert len(failed)==1 and failed[0]['numerical_failure'] is True


def test_global_completed_attestation_does_not_cover_interrupted_parent(fixture):
    row=fixture.fit(fixture.row());fixture.seal(row['output'],extra={'status':'running'})
    fixture.audit['all_recorded_completed_artifacts_verified']=True
    for record in fixture.audit['runs']:
        if record['directory']==row['output']:record.pop('all_artifacts_verified')
    verifier=em.Verifier(fixture.campaign,fixture.release,fixture.audit,fixture.mapper,{})
    with pytest.raises(ValueError,match='attest'):
        verifier.base(row['output'],completed=False)


@pytest.mark.parametrize('name',['evaluation/hidden.json','./evaluation/hidden.json','safe/../evaluation/hidden.json','../outside.json','alias.json'])
def test_parallel_hashing_prevalidates_all_paths_before_workers_or_label_reads(fixture,monkeypatch,name):
    row=fixture.fit(fixture.row());directory=fixture.mapper(row['output'])
    hidden=directory/'evaluation/hidden.json';hidden.parent.mkdir();hidden.write_text('synthetic forbidden evaluation bytes')
    outside=directory.parent/'outside.json';outside.write_text('synthetic outside bytes')
    if name=='alias.json':(directory/name).symlink_to(hidden)
    target=outside if name=='../outside.json' else hidden
    run=json.loads((directory/'run.json').read_text())
    # Append the malicious entry after valid artifacts: submission must not
    # begin while a later name remains unchecked.
    run['artifact_sha256'][name]=sha256(target);write_json(directory/'run.json',run)
    audit=next(r for r in fixture.audit['runs'] if r['directory']==row['output'])
    audit.update(run_sha256=sha256(directory/'run.json'),artifact_sha256=run['artifact_sha256'])
    original=em.sha256
    def guarded(path):
        if Path(path).resolve() in {hidden.resolve(),outside.resolve()}:pytest.fail('Forbidden bytes opened')
        return original(path)
    monkeypatch.setattr(em,'sha256',guarded)
    monkeypatch.setattr(em,'ThreadPoolExecutor',lambda **kwargs:pytest.fail('Hash workers created before full path validation'))
    verifier=em.Verifier(fixture.campaign,fixture.release,fixture.audit,fixture.mapper,{})
    with pytest.raises(ValueError):verifier.base(row['output'])


def test_parallel_hashing_preserves_order_and_existing_direct_rechecks(fixture,monkeypatch):
    row=fixture.fit(fixture.row());directory=fixture.mapper(row['output'])
    original=em.sha256;main=threading.get_ident();lock=threading.Lock();barrier=threading.Barrier(4)
    worker_calls=[];counts={}
    def observed(path):
        thread=threading.get_ident()
        with lock:
            key=str(path);counts[key]=counts.get(key,0)+1
            if thread!=main:worker_calls.append(thread);first_four=len(worker_calls)<=4
            else:first_four=False
        if first_four:barrier.wait(timeout=5)
        return original(path)
    monkeypatch.setattr(em,'sha256',observed)
    verifier=em.Verifier(fixture.campaign,fixture.release,fixture.audit,fixture.mapper,{})
    value=verifier.base(row['output'])
    assert len(set(worker_calls))==4
    assert value['locally_verified']==list(value['hashes'])
    assert value['remote_only']==[]
    assert counts[str(directory/'config.json')]>=3
    for name in em.SOURCE_FILES:assert counts[str(directory/'source'/name)]>=2


def test_parallel_hashing_still_rejects_tampered_artifact_before_labels(fixture,monkeypatch):
    row=fixture.fit(fixture.row());fixture.prediction(row)
    (fixture.mapper(row['output'])/'model.pt').write_bytes(b'tampered without changing trusted digest')
    monkeypatch.setattr(em,'load_evaluation_labels',lambda *args:pytest.fail('Labels accessed after integrity failure'))
    with pytest.raises(ValueError,match='Artifact hash mismatch'):fixture.evaluate(allow_incomplete=True)
    failed=json.loads((fixture.root/'analysis/FAILED.json').read_text())
    assert failed['stage']=='preflight' and failed['labels_may_have_been_loaded'] is False


def _auditor_fixture(fixture):
    row=fixture.fit(fixture.row())
    release=fixture.root/'audit_release';release.mkdir()
    campaign=copy.deepcopy(fixture.campaign)
    initial=campaign['initializations'][row['initial_index']]
    initial['output']=str(fixture.mapper(row['initial']))
    initial['manifest']=str(fixture.mapper(row['manifest']))
    campaign['output_root']=str(fixture.root/'outputs')
    cp=release/'campaign_manifest.json';write_json(cp,campaign)
    sources={'campaign_manifest.json':sha256(cp)}
    for name in em.SOURCE_FILES:
        p=release/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture '+name);sources[name]=sha256(p)
    transfer=fixture.root/'audit_transfer.json';write_json(transfer,dict(sources=sources,protocol_sha256=protocol_digest()))
    return cp,release,transfer,row


def test_reusable_completion_auditor_bundle_excludes_labels(fixture):
    from experiments.paper1_review.audit_completed import audit
    import tarfile
    cp,release,transfer,row=_auditor_fixture(fixture)
    result=audit(cp,sha256(cp),release,transfer,sha256(transfer),fixture.root/'remote_audit',phase='initialize',index=row['initial_index'],bundle=True)
    assert result['labels_loaded'] is False
    proof=json.loads((fixture.root/'remote_audit/completion_audit.json').read_text())
    assert proof['completed_run_count']==1 and proof['runs'][0]['all_artifacts_verified']
    anchor=json.loads((fixture.root/'remote_audit/release_hashes.json').read_text())
    assert anchor['source_file_sha256']==fixture.sources and anchor['protocol']==PROTOCOL
    with tarfile.open(fixture.root/'remote_audit/label_free_bundle.tar.gz') as tar:
        names=tar.getnames()
    assert not any('label' in n or '/evaluation/' in n for n in names)
    assert any(n.endswith('autoencoder.pt') for n in names)


def test_reusable_completion_auditor_rejects_changed_release(fixture):
    from experiments.paper1_review.audit_completed import audit
    cp,release,transfer,row=_auditor_fixture(fixture)
    (release/'experiments/paper1_review/models.py').write_text('changed source')
    with pytest.raises(ValueError,match='Frozen release changed'):
        audit(cp,sha256(cp),release,transfer,sha256(transfer),fixture.root/'bad_audit',phase='initialize',index=row['initial_index'])
