"""Optional local scheduler entry: new immutable cycle directory on every run.

Example cron/systemd invokes this explicitly; this script installs no schedule,
does not upload raw files, and never promotes candidates or connects to trading.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from letf.lab.operations import cycle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output-root',required=True,type=Path)
    a=p.parse_args()
    name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:10]
    result=cycle(json.loads(a.config.read_text()),output=a.output_root/name)
    print(json.dumps({'output':str(a.output_root/name),'status':result['status'],
                      'promoted':result['promoted'],'prospective_evaluated_months':result['prospective_evaluated_months']}))


if __name__=='__main__':main()
