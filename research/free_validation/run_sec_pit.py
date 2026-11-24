"""CLI: download SEC companyfacts using declared User-Agent; build PIT snapshot.

User must set SEC_USER_AGENT to a real contact email. This never rewrites a
sealed strategy holdout, and does not use historical values before filing date.
"""
from __future__ import annotations
import argparse
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import pandas as pd
from .sec_pit import get_companyfacts, facts_pit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cik', type=int, required=True)
    p.add_argument('--asof', default=date.today().isoformat())
    p.add_argument('--tag', default='Revenues')
    p.add_argument('--unit', default='USD')
    p.add_argument('--output', default='results/SEC_PIT')
    args=p.parse_args()
    if pd.Timestamp(args.asof) > pd.Timestamp(date.today()):
        raise ValueError('No future asof dates')
    content=get_companyfacts(args.cik)
    table=facts_pit(content,args.tag,args.asof,unit=args.unit)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    table.to_csv(out/'point_in_time_facts.csv',index=False)
    metadata={'source':'SEC companyfacts (public API)', 'cik':args.cik,
              'company':content.get('entityName'), 'tag':args.tag, 'asof':args.asof,
              'rows':len(table),'availability_date':'SEC filed date, not fiscal period end',
              'data_hash':sha256(json.dumps(content,sort_keys=True).encode()).hexdigest(),
              'limitations':['Not CRSP/Compustat','No delisting-safe universe',
                             'Revisions/filing time only resolved at daily precision',
                             'Not used to unseal any existing strategy holdout']}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
