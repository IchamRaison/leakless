import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from harness import metrics
root=Path('docs/evidence/tslm-v2/causal-d2-001'); run=root/'run'; reference=Path('docs/evidence/tslm-v2/train-diagnostic-001')
read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
pre=read(root/'preregistration.json');complete=read(run/'complete.json');summary=read(run/'summary.json');fold_document=read(reference/'folds.json');probes=read(reference/'probes.json')
assert digest({k:v for k,v in complete.items() if k!='receipt_sha256'})==complete['receipt_sha256']=='48ac9de1fef8238c616207f3eef557bd3c3fc137c26535a7762250bfec126747'
assert sha(root/'preregistration.json')==complete['preregistration_sha256']=='67ed6bd5a70caa7201d961fb692b0a0aa7eaccea53e3d41ee157fd0bc0ae8977'
assert {str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='complete.json'}==complete['artifacts_sha256']
assert sha(reference/'folds.json')=='5c2bea733f8f2a2dc525b9738a5aa40ae3ce220cf6d76d712cab984afb4d5efb'
assert sha(reference/'probes.json')=='aa16a7b1230ed0badbaac8db942b721ae07a3d213903645025bc2d80af456fc8'
assert sha(Path('manifests/split_v2.csv'))=='7a8716a35284434292314c10da58663e9f848be60edf18db0f98ef9d63d17896'
assert pre['folds']==fold_document['folds']
for relative,expected in pre['identity']['source_sha256'].items():assert sha(Path(relative))==expected,relative
for remote,expected in pre['identity']['files_sha256'].items():
 if '/code-causal-d2-303609e/' in remote:assert sha(Path(remote.split('/code-causal-d2-303609e/',1)[1]))==expected,remote
expected_params=dict(loss='log_loss',learning_rate=.05,max_iter=200,max_leaf_nodes=7,min_samples_leaf=10,l2_regularization=1,max_bins=255,early_stopping=False,random_state=20260913,class_weight=None,categorical_features=None)
assert pre['params']==expected_params
assert all(pre['runtime']['get_params'][k]==v for k,v in expected_params.items())
assert pre['runtime']['get_params']['warm_start'] is False
assert all(pool['num_threads']==1 for pool in pre['runtime']['threadpools_limited'])
assert pre['counts']==complete['counts']==summary['counts']=={'n_configs_compared':1,'n_fits':3,'reference_fits':0}
assert pre['fits_during_preregistration']==0 and pre['thread_limit']==1
assert pre['identity']['matrix_shape']==[598,256] and pre['identity']['matrix_dtype']=='float32' and pre['identity']['old_and_current_bands_byte_identical'] is True
assert complete['completed'] is True and complete['inputs_sources_runtime_unchanged'] is True
for k in ('limits','selection_policy','pooled_auc','diagnostic_threshold','official_validation_test_external_audio_or_cache_read','qwen_loaded'):assert pre[k]==summary[k]
with Path('manifests/split_v2.csv').open() as stream:manifest=[r for r in csv.DictReader(stream) if r['fold']=='train']
rows=sorted([{'clip_id':r['clip_id'],'group_id':r['group_id'],'fold':'train','label':'leak' if r['label']=='leak' else 'no_leak'} for r in manifest],key=lambda r:r['clip_id'])
assert digest(rows)==pre['identity']['rows_sha256'];byid={r['clip_id']:r for r in rows}
assert len(rows)==len(byid)==598 and len({r['group_id'] for r in rows})==102
assert sum(r['label']=='leak' for r in rows)==294
assert sorted(f['fold_id'] for f in pre['folds'])==[0,1,2]
seen_held=[];computed={};nll_max_error=0.;endpoints=[]

def checked_report(ids,probabilities,stored):
 global nll_max_error
 assert len(ids)==len(set(ids))==len(probabilities) and set(ids)==set(probabilities)
 rs=[byid[cid] for cid in ids];y=np.array([int(r['label']=='leak') for r in rs]);p=np.array([probabilities[cid] for cid in ids],dtype=np.float64);g=np.array([r['group_id'] for r in rs])
 assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 actual=metrics.evaluate(y,p,g,.5);_,gy,gp=metrics.aggregate_clusters(y,p,g);actual.update(group_roc_auc_full=metrics.roc_auc(gy,gp),clip_roc_auc_full=metrics.roc_auc(y,p))
 assert actual=={k:v for k,v in stored.items() if k not in ('binary_nll','fold_id','n_fit_clips')}
 nll=[];bad=[]
 for cid,target,score in zip(ids,y,p):
  if (target==1 and score==0) or (target==0 and score==1):nll.append(math.inf);bad.append(cid)
  else:nll.append(-math.log(float(score)) if target else -math.log1p(-float(score)))
 ninf=sum(math.isinf(n) for n in nll);saved=stored['binary_nll']
 assert saved['n_clips']==len(ids) and saved['n_infinite']==ninf
 if ninf:assert saved['mean'] is None and saved['sum'] is None
 else:
  total=math.fsum(nll);mean=total/len(nll);delta=max(abs(total-saved['sum']),abs(mean-saved['mean']));nll_max_error=max(nll_max_error,delta);assert delta<1e-12,delta
 actual['binary_nll']=saved
 return actual,{'scores_zero':int(np.sum(p==0)),'scores_one':int(np.sum(p==1)),'wrong_zero':int(np.sum((p==0)&(y==1))),'wrong_one':int(np.sum((p==1)&(y==0))),'n_infinite':ninf,'affected_groups':len({byid[cid]['group_id'] for cid in bad})}

for fold in pre['folds']:
 fid=fold['fold_id'];fit=fold['train_ids'];held=fold['heldout_ids'];assert len(fit)==len(set(fit)) and len(held)==len(set(held))
 assert not set(fit)&set(held) and set(fit)|set(held)==set(byid)
 fitgroups={byid[cid]['group_id'] for cid in fit};heldgroups={byid[cid]['group_id'] for cid in held}
 assert not fitgroups&heldgroups and sorted(heldgroups)==fold['heldout_groups']
 for ids in (fit,held):assert {byid[cid]['label'] for cid in ids}=={'leak','no_leak'}
 seen_held.extend(held);predictions=read(run/f'fold-{fid}-predictions.json');stored=read(run/f'fold-{fid}-metrics.json')
 assert len(predictions)==598 and len({p['clip_id'] for p in predictions})==598
 assert all(set(p)=={'clip_id','inner_fold','partition','probability_leak'} and p['inner_fold']==fid and p['partition'] in ('fit','heldout') for p in predictions)
 assert stored['fold_id']==fid and stored['n_iter']==200 and stored==next(r for r in summary['folds'] if r['fold_id']==fid)
 computed[fid]={}
 for part,ids in (('fit',fit),('heldout',held)):
  selected=[p for p in predictions if p['partition']==part];scores={p['clip_id']:p['probability_leak'] for p in selected};assert len(scores)==len(selected)==len(ids)
  actual,saturation=checked_report(ids,scores,stored['partitions'][part]);computed[fid][part]=actual
  print('D2_VERIFIED',fid,part,'clips',len(ids),'groups',actual['n_clusters'],'AUCclip',actual['clip_roc_auc_full'],'AUCgroup',actual['group_roc_auc_full'],'NLL',actual['binary_nll']['mean'],'confusion',{k:actual['clip_level'][k] for k in ('tp','fp','fn','tn')},'saturation',saturation)
 for name,ref in probes.items():
  selected=[p for p in ref['predictions_by_fold'] if p['inner_fold']==fid];scores={p['clip_id']:p['probability_leak'] for p in selected};assert len(scores)==len(selected)==len(held)
  saved=next(r for r in summary['references_heldout'][name] if r['fold_id']==fid);historic=next(r for r in ref['folds'] if r['fold_id']==fid)
  assert {k:v for k,v in saved.items() if k!='binary_nll'}==historic
  actual,saturation=checked_report(held,scores,saved);computed[fid][name]=actual
  if saturation['n_infinite']:endpoints.append((name,fid,saturation))
  print('REFERENCE_VERIFIED',name,fid,'AUCclip',actual['clip_roc_auc_full'],'AUCgroup',actual['group_roc_auc_full'],'NLL',actual['binary_nll']['mean'],'saturation',saturation)
assert sorted(seen_held)==sorted(byid)
for part in ('fit','heldout'):
 for key in ('group_roc_auc_full','clip_roc_auc_full'):assert float(np.mean([computed[i][part][key] for i in range(3)]))==summary['mean_auc_by_partition'][part][key]
for name,ref in probes.items():
 for key,field in (('mean_group_roc_auc','group_roc_auc_full'),('mean_clip_roc_auc','clip_roc_auc_full')):assert float(np.mean([computed[i][name][field] for i in range(3)]))==ref[key]
 deltas=[]
 for fid in range(3):
  a,b=computed[fid]['heldout'],computed[fid][name];nll,bnll=a['binary_nll']['mean'],b['binary_nll']['mean'];valid=nll is not None and bnll is not None
  value={'fold_id':fid,'group_auc_delta':a['group_roc_auc_full']-b['group_roc_auc_full'],'clip_auc_delta':a['clip_roc_auc_full']-b['clip_roc_auc_full'],'binary_nll_delta':nll-bnll if valid else None,'binary_nll_delta_defined':valid};deltas.append(value)
  assert value==next(r for r in summary['comparisons_heldout'][name]['folds'] if r['fold_id']==fid)
 assert float(np.mean([r['group_auc_delta'] for r in deltas]))==summary['comparisons_heldout'][name]['mean_group_auc_delta']
 assert float(np.mean([r['clip_auc_delta'] for r in deltas]))==summary['comparisons_heldout'][name]['mean_clip_auc_delta']
rule=summary['comparisons_heldout']['primary'];assert rule['rule']==pre['primary_rule']=='three_strictly_positive_heldout_group_auc_deltas_vs_TimeNet256'
assert rule['coherent_improvement_on_these_folds']==all(computed[i]['heldout']['group_roc_auc_full']>computed[i]['TimeNet256']['group_roc_auc_full'] for i in range(3))
assert rule['statistical_significance_or_independent_confirmation'] is False
print('D2_ALL_RECEIPT_COHORTS_SCORES_NLL_METRICS_REFERENCES_DELTAS_MEANS_CRITERION_PASS','independent_math_nll_max_sum_error',nll_max_error,'numpy',np.__version__)
print('SATURATION_WRONG_ENDPOINTS',endpoints)
print('MEANS',summary['mean_auc_by_partition'],'refs',[(name,v['mean_group_roc_auc'],v['mean_clip_roc_auc']) for name,v in probes.items()])
