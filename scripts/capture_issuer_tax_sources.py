"""Capture exact public 2024/25 tax records into a new local-only directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import requests
from scripts.build_multiyear_fund_bundle import OTHER_ISSUERS
from scripts.build_observed_etf_bundles import ici_rows


def capture(output,offline_cache=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    specs=[]
    for year in (2024,2025):
        specs.append((f'letf-ici-primary-{year}.xlsx',f'https://www.proshares.com/globalassets/proshares/documents/tax/ici_primary_{year}.xlsx','ProShares',year))
        for symbol in ('TQQQ','QLD','SSO','UPRO'):
            specs.append((f'letf-{symbol}-issuer-{year}.json',f'https://www.proshares.com/api/distributionsummary/?fund={symbol}&year={year}','ProShares',year))
    for issuer in OTHER_ISSUERS.values():
        for year,(name,url) in issuer['files'].items():specs.append((name,url,issuer['provider'],year))
    specs.append(('letf-ssga-ici-primary.xlsx','https://www.ssga.com/library-content/products/fund-docs/etfs/us/tax-documents/SPDR_ICI_Tax_Summary_Primary.xlsx','State Street',2025))
    def fetch(spec):
        name,url,provider,year=spec
        try:
            archived=datetime.now(timezone.utc).isoformat()
            if offline_cache is not None:blob=(Path(offline_cache)/name).read_bytes()
            else:
                r=requests.get(url,timeout=60);r.raise_for_status();blob=r.content
            p=output/name;p.write_bytes(blob)
            if name.endswith('.xlsx'):ici_rows(p,year=year)
            elif not isinstance(json.loads(blob),list) or not json.loads(blob):raise ValueError('Missing issuer cash rows')
            return name,{'path':name,'url':url,'provider':provider,'tax_year':year,'sha256':sha256(blob).hexdigest(),
                'archived_at':archived,'capture_mode':'local byte reuse' if offline_cache is not None else 'public HTTP download',
                'publication_vintage':'Dated final tax-year record; revisions remain possible','redistribution':'Not certified; local research only'},None
        except Exception as exc:return name,None,{'url':url,'status':'BLOCKED','reason':str(exc)}
    report={'files':{},'blocked':{},'scope':'Source capture only; complete tax/execution/account gates remain separate'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name,record,error in pool.map(fetch,specs):
            if error:report['blocked'][name]=error
            else:report['files'][name]=record
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['blocked']:raise SystemExit('Issuer capture incomplete; inspect explicit blocked records')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--offline-cache',type=Path)
    a=p.parse_args();r=capture(a.output,a.offline_cache);print(json.dumps({'verified_source_files':len(r['files'])}))
