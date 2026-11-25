"""Paired endpoint projections through exact bounded canonical-ledger maps.

This is NOT a full daily ledger replay of millions of paths. It memoizes the
canonical ledger's piecewise-affine terminal result ONLY for initial-buy/hold,
zero distributions/flows/sweep, single/federal/100k-income/frozen-2026 scenarios.
Drawdowns are rigorous bounds, not fabricated exact after-tax daily curves.
"""
from hashlib import sha256
import argparse,json
from pathlib import Path
import numpy as np
from letf.account_projection import CanonicalTerminalMap
from letf.macro.artifacts import canonical
from letf.macro.precision import quantile_interval,exact_probability
from letf.macro.precision_run import file_hash,json_safe


def project_accounts(precision_report,*,output):
    output=Path(output)
    if output.exists():raise ValueError('Existing account evidence cannot be overwritten')
    report=json.loads(Path(precision_report).read_text())
    if not report['passed']:raise ValueError('Underlying registered sampling precision did not pass')
    root=Path(precision_report).parent;arrays=[]
    for sample in report['samples']:
        path=root/sample['path']
        if file_hash(path)!=sample['sha256']:raise ValueError('Frozen paired sample checksum failed')
        arrays.append(np.memmap(path,dtype=sample['dtype'],mode='r',shape=tuple(sample['shape'])))
    spec=report['registration']['protocol'];keys=[(h,l) for h in spec['horizons'] for l in spec['leverages']]
    # The upstream path budget was selected from registered checkpoints using
    # these same paths. Protect every possible stopping point as well as every
    # account/metric: a fixed-look interval after optional stopping is not enough.
    looks=len(spec['checkpoints_per_seed'])
    rows=[];alpha=spec['family_alpha_quantiles']/(len(keys)*2*2*3*looks)
    palpha=spec['family_alpha_probabilities']/(len(keys)*2*3*looks)
    for j,(h,leverage) in enumerate(keys):
        logs=np.concatenate([a[:,j,0] for a in arrays]);dd=np.concatenate([a[:,j,1] for a in arrays])
        real_logs=np.concatenate([a[:,j,4] for a in arrays]);nav=np.exp(logs)
        if (nav<=0).any() or not np.isfinite(nav).all():raise ValueError('Zero/underflowed NAV requires actual dated liquidation; no path deletion')
        for account in ('taxable','roth'):
            lookup=CanonicalTerminalMap.build(account,float(nav.min()),float(nav.max()),horizon=h)
            net,taxes=lookup.evaluate(nav);real=net*np.exp(real_logs-logs)
            if (net<=0).any() or not np.isfinite(real).all():raise ValueError('Nonpositive/nonfinite account net assets; no clipping')
            intervals={}
            for name,values in [('nominal',net),('real',real)]:
                intervals[name]=[dict(quantile_interval(np.log(values),q,alpha),quantile=q) for q in (.1,.5,.9)]
            n=len(nav);slip=.0002;pre_sale_value=10000/(1+slip)*nav;terminal_drag=1-net/pre_sale_value
            # Without daily peak timing, give conservative pathwise bounds:
            # refund can lower a terminal-only trough; tax/sale costs can deepen
            # it. Never label underlying fund DD as an exact after-tax DD.
            lower=np.maximum(0.,dd-np.maximum(0.,-terminal_drag)-1e-12)
            upper=np.minimum(1.,dd+slip/(1+slip)+np.maximum(0.,terminal_drag)+1e-12)
            probs={name:exact_probability(int(mask.sum()),n,palpha) for name,mask in {
                'nominal_loss':net<10000.,'real_loss':real<10000.,'terminal_loss_over90':net<1000.}.items()}
            inflation=np.expm1((logs-real_logs)/h);sensitivity={}
            for label,mask in [('annualized_inflation_over4pct',inflation>.04),('annualized_inflation_under2pct',inflation<.02)]:
                sensitivity[label]={'paths':int(mask.sum()),'real_assets_p10_p50_p90':np.quantile(real[mask],[.1,.5,.9]).tolist() if mask.any() else None,
                    'scope':'Conditional joint-model subset, not identified causal inflation response'}
            rows.append({'horizon':h,'leverage':leverage,'account':account,'paired_paths':n,
                'net_assets_p10_p50_p90':np.quantile(net,[.1,.5,.9],method='inverted_cdf').tolist(),
                'real_net_assets_p10_p50_p90':np.quantile(real,[.1,.5,.9],method='inverted_cdf').tolist(),
                'lump_sum_cagr_p10_p50_p90':np.quantile(np.expm1(np.log(net/10000)/h),[.1,.5,.9],method='inverted_cdf').tolist(),
                'log_asset_sampling_intervals':intervals,
                'asset_precision_passed':all(q['maximum_error']<=.02 for v in intervals.values() for q in v),
                'after_tax_drawdown_percentile_bounds':list(zip(np.quantile(lower,[.1,.5,.9]).tolist(),np.quantile(upper,[.1,.5,.9]).tolist())),
                'probabilities_conditional':probs,'median_terminal_tax_payable':float(np.median(taxes)),
                'canonical_map_knots':lookup.knots.tolist(),'canonical_map_net_assets':lookup.net.tolist(),
                'maximum_independent_ledger_residual':lookup.maximum_independent_residual,'macro_sensitivity':sensitivity})
        print('paired account endpoint projection complete',h,leverage,flush=True)
    result={'evidence':'HYPOTHETICAL_STRESS','model_approved':False,'investment_certification':'BLOCKED',
        'precision_report_sha256':file_hash(precision_report),'projection_implementation_sha256':sha256(Path(__file__).read_bytes()+Path(__file__).parents[1].joinpath('account_projection.py').read_bytes()).hexdigest(),
        'source_catalog_sha256':report['registration']['source_catalog_sha256'],'paired_paths_per_combination':len(logs),
        'method':'Exact piecewise-affine memoization of canonical PortfolioLedger terminal executions and tax, independently reconciled inside every segment',
        'full_daily_account_replay_for_every_path':False,
        'family_error_control':{'possible_upstream_stopping_looks':looks,
            'quantile_interval_alpha':alpha,'probability_interval_alpha':palpha,
            'method':'Bonferroni across all accounts, nominal/real quantiles or probabilities, and every possible registered upstream stopping point'},
        'profile':{'initial_balance':10000.,'ordinary_income':100000.,'filing_status':'single','state':'FEDERAL_ONLY',
            'frozen_tax_year':2026,'distributions':0.,'contributions':0.,'cash_sweep':0.,'spread_bps':2.,'slippage_bps':1.,
            'Roth':'Existing balance; internal liquidation only, not personal withdrawal'},
        'all_account_asset_precision_gates_passed':all(r['asset_precision_passed'] for r in rows),'results':rows,
        'limitations':['Zero distributions is an explicit hypothetical scenario, not a forecast of actual TQQQ/UPRO taxable distributions',
            'Nonzero distributions/flows, MA, different income, unknown tax law or intermediate rebalancing require full canonical daily replay',
            'Account drawdown bounds include terminal tax/cost uncertainty in timing; exact after-tax daily drawdown is not verified',
            'Net assets include terminal tax/refund payable, not all immediately spendable cash',
            'Sampling intervals exclude model, historical source, parameter and future-tax-policy uncertainty',
            'No modeled liquidation in the finite training support does not bound unseen crash losses']}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(canonical(json_safe(result)));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--precision-report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=project_accounts(a.precision_report,output=a.output);print(r['all_account_asset_precision_gates_passed'])
