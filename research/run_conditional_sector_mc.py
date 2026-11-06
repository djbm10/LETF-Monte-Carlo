"""Reproducible CLI for joint-regime research; executes frozen policies, never promotes strategies."""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import date, timedelta
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from letf.simulation.sector_regime import SECTOR_PAIRS, Scenario, build_market_data, evaluate, manifest


def synthetic_fixture():
    """TESTING ONLY: labeled synthetic data to smoke-test workflow and deterministic seeds."""
    rng = np.random.default_rng(10)
    idx = pd.bdate_range('2018-01-01', periods=850)
    common = rng.normal(.00035, .011, len(idx))
    tech = common * 1.25 + rng.normal(0, .007, len(idx))
    out = {}
    ret = {'SPY': common, 'QQQ': tech}
    for j, (sector, lev) in enumerate(SECTOR_PAIRS.items()):
        u = common * (.7 + j * .04) + rng.normal(0, .009, len(idx))
        ret[sector] = u
        ret[lev] = 2 * u - .00028
    for name, returns in ret.items():
        out[name] = 100 * np.cumprod(1 + returns)
    out['^IRX'] = np.full(len(idx), 2.0)
    closes = pd.DataFrame(out, index=idx)
    opens = closes.drop(columns='^IRX').copy()
    vix = pd.Series(np.where(np.arange(len(idx)) % 260 < 80, 36.0, 17.0), index=idx)
    return closes, opens, vix


def get_data(args):
    if args.synthetic_smoke:
        return synthetic_fixture(), 'SYNTHETIC_FIXTURE_NO_RESEARCH_EVIDENCE'
    if args.fetch_yahoo:
        if not args.asof:
            raise ValueError('Use --asof YYYY-MM-DD for live Yahoo source: exclude incomplete sessions')
        try:
            import yfinance as yf
        except ImportError as e:
            raise RuntimeError('Install yfinance for --fetch-yahoo') from e
        names = sorted(set(['QQQ', 'SPY', '^IRX', '^VIX', *SECTOR_PAIRS.keys(), *SECTOR_PAIRS.values()]))
        result = {}
        fail = []
        end = (pd.Timestamp(args.asof) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        for i in range(0, len(names), 8):
            chunk = names[i:i + 8]
            d = yf.download(chunk, start='2005-01-01', end=end, group_by='ticker',
                            auto_adjust=True, threads=False, progress=False, timeout=40)
            for n in chunk:
                try:
                    sub = d[n]
                    result[(n, 'Open')] = pd.to_numeric(sub['Open'], errors='coerce')
                    result[(n, 'Close')] = pd.to_numeric(sub['Close'], errors='coerce')
                except Exception:
                    fail.append(n)
        if fail:
            raise ValueError(f'Yahoo source missing instruments: {sorted(set(fail))}')
        all_price = pd.DataFrame(result).sort_index()
        closes = pd.DataFrame({n: all_price[(n, 'Close')] for n in names if n != '^VIX'})
        opens = pd.DataFrame({n: all_price[(n, 'Open')] for n in names if n not in ('^VIX', '^IRX')})
        vix = all_price[('^VIX', 'Close')]
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        closes.to_csv(out / 'source_adjusted_close.csv')
        opens.to_csv(out / 'source_adjusted_open.csv')
        vix.to_csv(out / 'source_vix.csv')
        return (closes, opens, vix), 'YAHOO_ADJUSTED_OHLC_DOWNLOAD'
    if not args.close or not args.open or not args.vix:
        raise ValueError('Supply --close --open --vix or --fetch-yahoo, or --synthetic-smoke')
    c = pd.read_csv(args.close, index_col=0, parse_dates=True)
    o = pd.read_csv(args.open, index_col=0, parse_dates=True)
    vi = pd.read_csv(args.vix, index_col=0, parse_dates=True)
    if '^VIX' in vi:
        v = vi['^VIX']
    elif vi.shape[1] == 1:
        v = vi.iloc[:, 0]
    else:
        raise ValueError('VIX input must include ^VIX or exactly one value column')
    return (c, o, v), 'USER_SUPPLIED_ADJUSTED_OHLC'


def parse_csv_values(s, parser):
    return [parser(x.strip()) for x in s.split(',') if x.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--fetch-yahoo', action='store_true')
    source.add_argument('--synthetic-smoke', action='store_true')
    p.add_argument('--close');p.add_argument('--open');p.add_argument('--vix')
    p.add_argument('--asof', help='Last fully completed session; required for Yahoo download')
    p.add_argument('--output', default='results/conditional_sector_research')
    p.add_argument('--years', default='3,10,20')
    p.add_argument('--blocks', default='21,63,126')
    p.add_argument('--modes', default='current,unconditional,persistent_high')
    p.add_argument('--transition-modes', default='semi_markov,block_markov')
    p.add_argument('--paths', type=int, default=200)
    p.add_argument('--seed', type=int, default=20261106)
    p.add_argument('--high-duration-multipliers', default='1.0')
    p.add_argument('--funding-spread-bps', default='100')
    p.add_argument('--max-scenarios', type=int, default=150)
    args = p.parse_args(argv)
    data, provenance = get_data(args)
    market = build_market_data(*data, asof=args.asof, min_source_days=252)
    grid = list(itertools.product(
        parse_csv_values(args.years, int), parse_csv_values(args.blocks, int),
        parse_csv_values(args.modes, str), parse_csv_values(args.transition_modes, str),
        parse_csv_values(args.high_duration_multipliers, float),
        parse_csv_values(args.funding_spread_bps, float),
    ))
    if len(grid) > args.max_scenarios:
        raise ValueError(f'{len(grid)} scenarios exceed --max-scenarios={args.max_scenarios}; raise only deliberately')
    out = Path(args.output);out.mkdir(parents=True, exist_ok=True)
    configs, rows = [], []
    for i, (yr, bl, mode, tm, multiplier, spread) in enumerate(grid):
        c = Scenario(years=yr, paths=args.paths, block_days=bl, mode=mode,
                     transition_mode=tm, high_duration_multiplier=multiplier,
                     funding_spread_bps_annual=spread, seed=args.seed + i * 7919)
        configs.append(c)
        row = evaluate(market, c)
        row['scenario_id'] = i
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'scenario_results.csv', index=False)
        print(f'SCENARIO {i+1}/{len(grid)} {mode}/{tm} {yr}y {bl}d '
              f'blend_wealthier={row["p_blend_higher_terminal_wealth"]:.1%}', flush=True)
    details = manifest(market, configs)
    details['source_provenance'] = provenance
    (out / 'manifest.json').write_text(json.dumps(details, indent=2))
    print(f'COMPLETE {len(rows)} scenarios; sample={market.source_start}–{market.source_end} '
          f'asof={market.asof}; output={out}', flush=True)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
