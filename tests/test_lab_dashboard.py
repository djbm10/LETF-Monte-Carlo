import copy
import http.client
import json
import threading
import time

import pytest

from letf.lab.server import LabServer,Jobs
from letf.lab.html import document,render_report,report_view
from test_lab_portfolio import request


def test_loopback_dashboard_runs_real_fixture_ledger_and_protects_inputs(tmp_path):
    profile=request(tmp_path);server=LabServer({'fixture':profile},tmp_path/'jobs',port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    host,port=server.server_address
    def call(path,body=None,token=True,origin=None,host_header=None):
        client=http.client.HTTPConnection(host,port,timeout=10)
        headers={'Content-Type':'application/json'}
        if token:headers['X-Lab-Token']=server.token
        if origin:headers['Origin']=origin
        if host_header:headers['Host']=host_header
        client.request('POST' if body is not None else 'GET',path,body=json.dumps(body) if body is not None else None,headers=headers)
        response=client.getresponse();raw=response.read();client.close()
        return response.status,raw
    try:
        code,html=call('/');assert code==200 and b'LETF Research Laboratory'in html
        assert call('/api/profiles',token=False)[0]==403
        assert call('/api/profiles',origin='https://evil.example')[0]==403
        assert call('/api/profiles',host_header='evil.example')[0]==403
        bad=copy.deepcopy(profile);bad['bundles']['X']='/etc'
        assert call('/api/run',dict(profile='fixture',request=bad))[0]==400
        code,value=call('/api/run',dict(profile='fixture',request=profile))
        assert code==200;key=json.loads(value)['id']
        deadline=time.monotonic()+15
        while True:
            _,raw=call('/api/jobs/'+key);job=json.loads(raw)
            if job['status']in ('DONE','FAILED','BLOCKED'):break
            assert time.monotonic()<deadline
            time.sleep(.03)
        assert job['status']=='DONE',job
        assert job['view']['report']['classification']=='SOFTWARE_FIXTURE'
        assert job['view']['report']['accounts']['roth']['tax_paid']==0
        exported=tmp_path/'jobs'/key/'report.html';assert exported.exists()
        assert '<script src='not in exported.read_text()
        (tmp_path/'jobs'/key/'taxable_curve.csv').write_text('tampered')
        with pytest.raises(ValueError,match='checksum'):report_view(tmp_path/'jobs'/key/'report.json')
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)


def test_contribution_generator_requires_separate_eligibility_and_never_double_counts_json(tmp_path):
    profile=request(tmp_path);jobs=Jobs({'fixture':profile},tmp_path/'jobs')
    try:
        body=dict(profile='fixture',request=profile,annual_contribution=3000.)
        with pytest.raises(ValueError,match='MAGI'):jobs.prepare(body)
        body.update(annual_roth_magi=120000.,annual_eligible_compensation=90000.)
        generated=jobs.prepare(body)
        assert sum(f['amount'] for f in generated['cash_flows'])==6000.
        assert [f['date'] for f in generated['cash_flows']]==['2024-01-02','2025-01-02']
        assert all(f['magi']==120000. and f['compensation']==90000. for f in generated['cash_flows'])
        assert jobs.prepare(dict(body,annual_contribution=0.))['cash_flows']==[]
    finally:jobs.close()


def test_dashboard_cancellation_writes_no_completed_performance(tmp_path,monkeypatch):
    import letf.lab.server as module
    profile=request(tmp_path);entered=threading.Event();released=threading.Event()
    def slow(request,*,output,progress):
        entered.set();assert released.wait(timeout=5)
        progress({'session':0,'total_sessions':1,'completed_replays':0,'total_replays':1})
        raise AssertionError('Cancellation should interrupt before any result')
    monkeypatch.setattr(module,'run_observed',slow)
    jobs=Jobs({'fixture':profile},tmp_path/'jobs')
    try:
        key=jobs.submit(dict(profile='fixture',request=profile))['id']
        assert entered.wait(timeout=5);jobs.cancel(key);released.set()
        deadline=time.monotonic()+5
        while jobs.get(key)['status']!='CANCELLED':
            assert time.monotonic()<deadline;time.sleep(.01)
        assert not (tmp_path/'jobs'/key/'report.json').exists()
        receipt=json.loads((tmp_path/'jobs'/key/'cancelled.json').read_text())
        assert not receipt['completed_performance_result']
    finally:released.set();jobs.close()


def test_report_html_escapes_untrusted_script_labels_without_remote_resources():
    attack='</script><script>alert("private data")</script>'
    html=document({'report':{'classification':'SOFTWARE_FIXTURE','label':attack},'curves':{}})
    assert attack not in html and '\\u003c/script'in html
    assert '<script src='not in html and 'https://cdn.'not in html
