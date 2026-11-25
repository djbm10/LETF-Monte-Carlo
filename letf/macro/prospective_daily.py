"""Public aggregate forecast commitments and genuine-future-only assessment.

Only existing simple bootstrap references. Full ensembles and source snapshots
stay local under their licenses. Compact quantile/tail forecasts can be committed
to Git before outcomes and assessed even after loss of a local cache.
"""
from datetime import datetime,timezone
from hashlib import sha256
import json,os,tarfile,io
from pathlib import Path,PurePosixPath
import numpy as np
import pandas as pd
from letf.macro.artifacts import ArtifactStore,canonical,fingerprint
from letf.macro.data import CORE,train_slice
from letf.macro.models import MacroModel
from letf.macro.resampling import sample_with_lineage
from letf.macro.daily_tails import realized_monthly_variance,training_templates,daily_bridge,longest_underwater
from letf.macro.scoring import drawdowns,ensemble_crps

QUANTILES=[.05,.1,.25,.5,.75,.9,.95]


def utc_now():return datetime.now(timezone.utc)


def weighted_interval_score(quantiles,observation):
    """Proper WIS for three central intervals plus median, no fitted score."""
    x=np.asarray(quantiles,float);y=float(observation)
    if x.shape!=(7,) or not np.isfinite(x).all() or not np.isfinite(y) or (np.diff(x)<0).any():raise ValueError('Ordered seven-quantile forecast required')
    total=.5*abs(y-x[3])
    for alpha,i,j in ((.1,0,6),(.2,1,5),(.5,2,4)):
        total+=alpha/2*(x[j]-x[i]+2/alpha*max(x[i]-y,0)+2/alpha*max(y-x[j],0))
    return float(total/3.5)


def issue_daily(sources,*,registry,output,paths=2000,months=12,seed=20261129):
    output=Path(output)
    if output.exists():raise ValueError('Forecast commitments cannot be overwritten')
    now=utc_now();panel,meta=sources.panel();daily=sources.daily_factors()
    if daily.index.max().date()>now.date():raise ValueError('Future source observations')
    for name in meta['source_hashes']:
        if pd.Timestamp(sources.catalog['sources'][name]['retrieved_at'])>pd.Timestamp(now):raise ValueError('Future source capture')
    logs=np.log1p(daily['Mkt-RF']+daily.RF)
    panel['equity_log_variance']=realized_monthly_variance(logs).reindex(panel.index)
    frame=train_slice(panel,now.date(),480,120,allow_missing_transitions=True)
    store=ArtifactStore(registry);origin=pd.Timestamp(now.date())+pd.offsets.MonthEnd(0)
    gap=(origin.to_period('M')-frame.index[-1].to_period('M')).n
    dates=pd.date_range(origin+pd.offsets.MonthEnd(1),periods=months,freq='ME')
    templates,lengths=training_templates(logs.loc[frame.index[0].to_period('M').start_time:frame.index[-1]],frame)
    code=sha256(Path(__file__).read_bytes()+Path(__file__).with_name('resampling.py').read_bytes()+Path(__file__).with_name('daily_tails.py').read_bytes()).hexdigest()
    spec={'block_months':3,'minimum_bootstrap_months':120,'allow_incomplete_training':True,'stationary_mean_months':12}
    models=[]
    for j,kind in enumerate(('joint_block','stationary_block')):
        model=MacroModel.fit(kind,frame,spec)
        mid=store.put('models',{'kind':kind,'columns':model.columns,'parameters':model.parameters,
            'source_metadata':meta,'training_sha256':fingerprint(frame),'implementation_sha256':code},
            {'training':model.train_values,'templates':templates,'lengths':lengths})
        monthly,lineage=sample_with_lineage(model,months+gap,paths,seed+j);monthly=monthly[:,gap:];lineage=lineage[:,gap:]
        generated=daily_bridge(monthly,lineage,model.columns,templates,lengths,seed+100+j)
        # Today's snapshot is legitimately available for today's issue, but is
        # NOT thereby a historical decision-time vintage for old hindcasts.
        fid=store.put('forecasts',{'kind':kind,'model_id':mid,'issued_at':now.isoformat(),
            'dates':[str(d.date()) for d in dates],'paths':paths,'seed':seed+j,'daily_seed':seed+100+j,
            'implementation_sha256':code,'status':'PROSPECTIVE_RESEARCH_ONLY','clock':'21 synthetic sessions/month'},
            {'monthly':monthly,'daily_log':generated,'lineage':lineage})
        marginal=np.quantile(monthly,QUANTILES,axis=0).transpose(1,2,0)
        daily_quantiles=[]
        for t in range(months):daily_quantiles.append(np.quantile(generated[:,t*21:(t+1)*21].ravel(),QUANTILES).tolist())
        prefixes=[]
        for count in [n for n in (1,3,12) if n<=months]:
            sample=generated[:,:count*21]
            prefixes.append({'months':count,'max_drawdown_p50_p95_p99':np.quantile(drawdowns(sample),[.5,.95,.99]).tolist(),
                'underwater_sessions_p50_p95':np.quantile(longest_underwater(sample),[.5,.95]).tolist(),
                'crash_rate_per252_p50_p95':{str(c):np.quantile((sample<np.log1p(c)).sum(1)*252/sample.shape[1],[.5,.95]).tolist() for c in (-.05,-.1)}})
        models.append({'kind':kind,'forecast_id':fid,'monthly_quantiles':marginal.tolist(),'daily_marginal_quantiles':daily_quantiles,'prefix_tails':prefixes})
    commitment={'issued_at':now.isoformat(),'information_asof':str(now.date()),'origin':str(origin.date()),
        'outcome_dates':[str(d.date()) for d in dates],'quantile_levels':QUANTILES,'columns':list(frame.columns),
        'common_training_scale':frame.std(ddof=1).tolist(),'training_start':str(frame.index[0].date()),'training_end':str(frame.index[-1].date()),
        'unobserved_state_bridge_months':gap,'paths':paths,'source_catalog_sha256':sources.catalog_hash,
        'implementation_sha256':code,'models':models,'evidence':'HYPOTHETICAL_STRESS','model_approved':False,
        'assessment_rule':'WIS/common training scale; daily marginal interval and extreme-loss calibration after complete genuine future months only',
        'licensing':'Only aggregate derived forecasts public; raw licensed inputs and full ensembles stay in local sealed snapshot',
        'uncertainty':'Forecast percentiles are model distributions, not confidence intervals or validated real-world probabilities'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:f.write(canonical(commitment));f.flush();os.fsync(f.fileno())
    return commitment


def assess_daily(sources,commitment,*,output):
    output=Path(output)
    if output.exists():raise ValueError('Assessments cannot be overwritten')
    f=json.loads(Path(commitment).read_text());now=utc_now();issued=pd.Timestamp(f['issued_at'])
    first=pd.Timestamp(f['outcome_dates'][0]).to_period('M').start_time.tz_localize('UTC')
    if issued>=first or issued>pd.Timestamp(now):raise ValueError('Forecast not frozen before future outcome')
    dates=pd.to_datetime(f['outcome_dates'])
    due=[d for d in dates if d.date()<now.date()]
    if not due:
        result={'status':'PENDING_GENUINE_FUTURE_OUTCOMES','promoted':False,'evaluated_months':0}
    else:
        actual,meta=sources.panel();daily=sources.daily_factors()
        if daily.index.max().date()>now.date():raise ValueError('Future-dated observations cannot be assessed')
        eq=np.log1p(daily['Mkt-RF']+daily.RF);rows=[]
        for t,day in enumerate(dates):
            if day not in due:continue
            if day not in actual.index or not np.isfinite(actual.loc[day,list(CORE)]).all():continue
            observed=eq.loc[day.to_period('M').start_time:day]
            if observed.empty:continue
            for model in f['models']:
                losses={c:weighted_interval_score(model['monthly_quantiles'][t][f['columns'].index(c)],actual.loc[day,c])/
                    f['common_training_scale'][f['columns'].index(c)] for c in CORE}
                q=np.array(model['daily_marginal_quantiles'][t]);truth=observed.to_numpy()
                rows.append({'kind':model['kind'],'date':str(day.date()),'normalized_wis':losses,
                    'daily_90_interval_coverage':float(np.mean((truth>=q[0])&(truth<=q[-1]))),
                    'daily_lower5_breach_rate':float(np.mean(truth<q[0])),
                    'observed_sessions':len(observed),'observed_crashes_minus5':int((truth<np.log(.95)).sum())})
        result={'status':'ASSESSED_ELAPSED_OUTCOMES_ONLY' if rows else 'PENDING_VERIFIED_OUTCOME_DATA',
            'evaluated_rows':rows,'source_catalog_sha256':sources.catalog_hash,'promoted':False,
            'reason':'These reference diagnostics cannot approve a candidate or reuse historical rejected evidence'}
    result['commitment_sha256']=sha256(Path(commitment).read_bytes()).hexdigest();result['assessed_at']=now.isoformat()
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(canonical(result));return result


def seal_snapshot(sources,registry,commitment,archive):
    """Portable verified local archive; no claim of off-host/WORM storage."""
    archive=Path(archive)
    if archive.exists():raise ValueError('Archives are immutable')
    registry=Path(registry);files={'commitment.json':Path(commitment)}
    files['sources/catalog.json']=sources.folder/'catalog.json'
    for name,item in sources.catalog['sources'].items():
        sources.bytes(name)  # enforce provider hash before sealing
        files['sources/'+item['path']]=sources.folder/item['path']
    for p in registry.rglob('*'):
        if p.is_file() and '.pending' not in p.parts and p.name!='.lock':
            if p.suffix not in ('.json','.jsonl','.npz'):raise ValueError('Unreviewed registry file; never archive credentials')
            if p.is_symlink():raise ValueError('No registry symlinks')
            files['registry/'+str(p.relative_to(registry))]=p
    manifest={name:sha256(p.read_bytes()).hexdigest() for name,p in files.items()}
    archive.parent.mkdir(parents=True,exist_ok=True);pending=archive.with_name(archive.name+'.pending')
    with pending.open('xb') as raw:
        with tarfile.open(fileobj=raw,mode='w:gz') as tar:
            for name,path in sorted(files.items()):tar.add(path,arcname=name,recursive=False)
            blob=canonical({'files':manifest,'commitment_sha256':manifest['commitment.json'],'scope':'Local archival copy; off-host backup not verified'})
            info=tarfile.TarInfo('archive-manifest.json');info.size=len(blob);tar.addfile(info,io.BytesIO(blob))
        raw.flush();os.fsync(raw.fileno())
    os.rename(pending,archive)
    return verify_snapshot(archive)


def verify_snapshot(archive):
    with tarfile.open(archive,'r:gz') as tar:
        names=tar.getnames()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive members')
        for member in tar.getmembers():
            path=PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe archive member')
        manifest=json.load(tar.extractfile('archive-manifest.json'))
        if set(names)!={'archive-manifest.json',*manifest['files']}:raise ValueError('Archive inventory mismatch')
        for name,expected in manifest['files'].items():
            h=sha256()
            with tar.extractfile(name) as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=expected:raise ValueError('Archive source/forecast checksum failed')
    return {'status':'VERIFIED_LOCAL_ARCHIVE','files':len(manifest['files']),'commitment_sha256':manifest['commitment_sha256'],
        'off_host_backup_verified':False,'immutable_storage_verified':False}


def main():
    import argparse
    from letf.macro.data import HistoricalSources
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['issue','assess','seal','verify'])
    p.add_argument('--sources',type=Path);p.add_argument('--registry',type=Path);p.add_argument('--commitment',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.command=='verify':r=verify_snapshot(a.output)
    elif a.command=='issue':r=issue_daily(HistoricalSources(a.sources),registry=a.registry,output=a.output)
    elif a.command=='assess':r=assess_daily(HistoricalSources(a.sources),a.commitment,output=a.output)
    else:r=seal_snapshot(HistoricalSources(a.sources),a.registry,a.commitment,a.output)
    print(json.dumps({k:v for k,v in r.items() if k in ('status','issued_at','source_catalog_sha256','files','promoted','off_host_backup_verified')},indent=2))


if __name__=='__main__':main()
