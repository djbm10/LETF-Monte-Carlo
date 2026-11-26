"""Immutable local refresh/assessment/candidate cycle, always review-only."""
import json
from pathlib import Path

from letf.accuracy import UnsupportedModelError
from letf.lab.catalog import SourceLake,file_hash,inside_file,private_bytes,private_json
from letf.lab.public_sources import refresh
from letf.macro.data import HistoricalSources
from letf.macro.learning import ModelRegistry
from letf.macro.prospective_daily import assess_daily
from letf.macro.retraining import issue_candidates,evaluate_pending


def materialize_sources(original,lake,updates,target):
    """New compatible historical catalog. Old bytes/timestamps stay unchanged."""
    target=Path(target);target.mkdir(parents=True,exist_ok=False,mode=0o700)
    source=HistoricalSources(original);catalog=json.loads(json.dumps(source.catalog))
    catalog['parent_catalog_sha256']=source.catalog_hash
    catalog['materialized_from_private_source_lake']=True
    for name,item in catalog['sources'].items():
        blob,_=source.bytes(name)
        if name in updates:
            record,raw=lake.load(updates[name]);blob=raw.read_bytes()
            if record['name']!=name:raise ValueError('Refreshed series identity does not match training variable')
            item=dict(record['metadata']);item['sha256']=record['raw_sha256']
            item['path']=name+'.'+item.get('format','csv')
            item['source_lake_snapshot']=updates[name];catalog['sources'][name]=item
        # Retain nesting only when already reviewed, never follow symlinks.
        destination=target/item['path']
        if not destination.resolve().is_relative_to(target.resolve()):raise ValueError('Source path escapes materialized catalog')
        private_bytes(destination,blob)
    if set(updates)-set(catalog['sources']):
        raise ValueError('Refresh introduces unregistered training series; register a new source-schema version first')
    private_json(target/'catalog.json',catalog)
    check=HistoricalSources(target)
    for name in check.catalog['sources']:check.bytes(name)
    return check


def cycle(config, *, output):
    allowed={'schema','sources','registry','protocol','commitment','commitment_sha256','refresh',
             'issue_candidates','require_existing_registry','new_registry_intent','materialize_refreshed_training'}
    if set(config)-allowed or config.get('schema')!=1:raise ValueError('Unknown operations-cycle schema/fields')
    output=Path(output);output.mkdir(parents=True,exist_ok=False,mode=0o700)
    sources=HistoricalSources(config['sources']);original_hash=sources.catalog_hash
    if file_hash(config['commitment'])!=config['commitment_sha256']:
        raise ValueError('Frozen prospective commitment checksum mismatch')
    refresh_result=None
    if config.get('refresh'):
        r=config['refresh']
        if set(r)!={'lake','names'}:raise ValueError('Explicit source lake/names required')
        lake=SourceLake(r['lake']);refresh_result=refresh(lake,r['names'])
        private_json(output/'refresh.json',refresh_result)
        if config.get('materialize_refreshed_training'):
            if not refresh_result['complete']:
                raise UnsupportedModelError('Required refreshed training inputs incomplete; original sources not mutated')
            updates={name:row['snapshot'] for name,row in refresh_result['sources'].items()}
            sources=materialize_sources(config['sources'],lake,updates,output/'training-sources')
    assessment=assess_daily(sources,config['commitment'],output=output/'prospective-assessment.json')
    registry=ModelRegistry(config['registry']);protocol=json.loads(Path(config['protocol']).read_text())
    require_existing=config.get('require_existing_registry',True)
    if not require_existing and config.get('new_registry_intent')!='BEGIN_CURRENT_DATE_ONLY':
        raise ValueError('New registry needs explicit current-date-only intent; cache loss cannot reset old evidence')
    issuance={'status':'NOT_REQUESTED','promoted':False}
    if config.get('issue_candidates'):
        issuance=issue_candidates(sources,registry,protocol,output=output/'candidate-issue',require_existing=require_existing)
    evaluation=evaluate_pending(sources,registry,protocol,output=output/'candidate-assessment',allow_promotion=False)
    if evaluation['promoted']:raise AssertionError('Review-only operations cannot promote models')
    from letf.research_backup import create_bundle,restore_bundle
    roots={'source_snapshot':sources.folder,'commitment':Path(config['commitment']),'protocol':Path(config['protocol'])}
    if registry.root.exists():roots['candidate_registry']=registry.root
    receipt=create_bundle(roots,output/'cycle-research.tar.gz',provenance={
        'source_catalog_sha256':sources.catalog_hash,'commitment_sha256':config['commitment_sha256'],
        'purpose':'Private local current-prospective cycle, no off-host transfer'})
    recovered=output/'recovery-drill'
    recovery=restore_bundle(output/'cycle-research.tar.gz',receipt['archive_sha256'],recovered)
    restored=HistoricalSources(recovered/'source_snapshot')
    if restored.catalog_hash!=sources.catalog_hash:raise AssertionError('Recovered source catalog identity differs')
    for name in restored.catalog['sources']:restored.bytes(name)
    if issuance.get('forecast_ids'):
        restored_registry=ModelRegistry(recovered/'candidate_registry')
        for fid in issuance['forecast_ids']:restored_registry.store.load('forecasts',fid)
    report={'schema':1,'classification':'CONDITIONAL_STRESS','status':'LOCAL_RESEARCH_CYCLE_COMPLETED',
        'source_catalog_sha256':sources.catalog_hash,'original_catalog_sha256':original_hash,
        'protocol_sha256':file_hash(config['protocol']),'commitment_sha256':config['commitment_sha256'],
        'prospective_assessment_status':assessment['status'],'prospective_evaluated_months':assessment.get('evaluated_months',0),
        'refresh':refresh_result,'candidate_issue':issuance,'candidate_evaluation':evaluation,
        'promotion_mode':'REVIEW_ONLY','model_approved':False,'promoted':False,
        'original_forecasts_changed':False,'off_host_transfer_attempted':False,
        'local_archive':receipt,'local_recovery':recovery,
        'next_run':'Invoke again into a NEW output directory after newly verified public data/releases arrive; no background execution implied'}
    private_json(output/'operations-report.json',report)
    return report
