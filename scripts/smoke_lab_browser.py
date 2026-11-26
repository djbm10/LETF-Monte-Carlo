"""Optional real-browser conformance on authored fixtures, loopback only.

Requires separately installed Playwright; not a simulator runtime dependency.
No real/private provider profile is accepted or uploaded. All subprocesses,
server threads and jobs are closed before this script returns.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from letf.lab.catalog import private_json,file_hash
from letf.lab.fixtures import demo
from letf.lab.server import LabServer


def smoke(output):
    from playwright.sync_api import sync_playwright
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False,mode=0o700)
    demo(output/'fixture');profiles=json.loads((output/'fixture/profiles.json').read_text())
    if any(p['mode']!='fixture' for p in profiles.values()):raise ValueError('Only authored software fixtures')
    server=LabServer(profiles,output/'jobs',port=0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    errors=[];external=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True)
            try:
                page=browser.new_page(viewport={'width':1280,'height':1000})
                def restrict(route):
                    url=urlsplit(route.request.url)
                    if url.hostname=='127.0.0.1' and url.port==server.server_port:route.continue_()
                    else:external.append(route.request.url);route.abort()
                page.route('**/*',restrict);page.on('pageerror',lambda error:errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle')
                page.wait_for_function("document.querySelector('#profile').options.length>0")
                page.click('#run')
                page.wait_for_function("['DONE','BLOCKED','FAILED'].some(s=>document.querySelector('#status').textContent.startsWith(s))",timeout=120000)
                assert page.locator('#status').inner_text()=='DONE',page.locator('#status').inner_text()
                assert page.locator('#classification').inner_text()=='SOFTWARE_FIXTURE'
                assert page.locator('#table tr').count()==3
                assert page.locator('#chart polyline').count()==2
                page.screenshot(path=str(output/'desktop.png'),full_page=True)
                page.select_option('#account','roth');assert page.locator('#chart polyline').count()==1
                assert page.locator('#table tr').count()==2
                page.select_option('#curve','wealth')
                with page.expect_download() as item:page.click('#download-report')
                item.value.save_as(str(output/'downloaded-report.json'))
                report=json.loads((output/'downloaded-report.json').read_text())
                assert report['classification']=='SOFTWARE_FIXTURE' and len(report['accounts'])==2
                page.fill('#contribution','500');page.fill('#magi','100000');page.fill('#compensation','100000')
                with page.expect_download() as item:page.click('#download-config')
                item.value.save_as(str(output/'downloaded-request.json'))
                request=json.loads((output/'downloaded-request.json').read_text())
                assert [(f['date'],f['amount']) for f in request['cash_flows']]==[('2024-01-02',500.),('2025-01-02',500.)]
                page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(output/'mobile.png'),full_page=True)
                page.select_option('#state','MA');page.click('#run')
                page.wait_for_function("document.querySelector('#status').textContent.includes('MA')")
                assert page.locator('#table tr').count()==0 # failed request must not display stale wealth
                assert page.locator('#download-report').is_disabled()
                assert not errors,errors
                assert not external,external
                result={'classification':'SOFTWARE_FIXTURE','playwright':version('playwright'),'chromium':browser.version,
                    'completed_comparison':True,'account_filter':True,'curve_selection':True,'report_download':True,
                    'generated_contribution_config_download':True,'unsupported_state_rejected_without_stale_result':True,
                    'page_errors':errors,'external_page_requests':external,'browser_and_server_closed_on_return':True,
                    'screenshots':{n:file_hash(output/n) for n in ('desktop.png','mobile.png')}}
            finally:browser.close()
    finally:server.shutdown();server.server_close();thread.join(timeout=5)
    private_json(output/'browser-conformance.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(smoke(a.output)))
