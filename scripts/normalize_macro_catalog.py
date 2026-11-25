"""New metadata edition, verifying/copying immutable observations unchanged."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.capture_macro_sources import source_units
from letf.macro.data import HistoricalSources,french_daily


def normalize(source,output):
    old=HistoricalSources(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    catalog=json.loads(json.dumps(old.catalog))
    catalog['metadata_revision']='Explicit observation units and genuine source bounds; no return/rate observations changed'
    catalog['parent_catalog_sha256']=old.catalog_hash
    for name,meta in catalog['sources'].items():
        blob,_=old.bytes(name);(output/meta['path']).write_bytes(blob);meta.update(source_units(name))
        if meta['path'].endswith('.zip'):
            frame=french_daily(blob,date_digits=6 if name=='ff_monthly' else 8)
            meta.update(first_observation=str(frame.index[0].date()),last_observation=str(frame.index[-1].date()),
                observations=len(frame),missing_observations=int(frame.isna().to_numpy().sum()))
    (output/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n');return catalog


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();normalize(a.sources,a.output)
