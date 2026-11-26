"""Loopback-only laboratory dashboard. No shell, trading or external deployment."""
from concurrent.futures import ThreadPoolExecutor, CancelledError
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import threading
import uuid

import numpy as np
import pandas as pd

from letf.lab.catalog import private_json
from letf.lab.html import document, report_view, render_report
from letf.lab.portfolio import run_observed, validate_request, load_portfolio


class Jobs:
    def __init__(self,profiles,root):
        if not isinstance(profiles,dict) or not profiles:raise ValueError('Register explicit local input profiles')
        for request in profiles.values():validate_request(request)
        self.profiles=copy.deepcopy(profiles);self.root=Path(root)
        if self.root.is_symlink():raise ValueError('Job root cannot be a symlink')
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if self.root.stat().st_mode&0o077:raise ValueError('Private job root needs0700 permissions')
        self.jobs={};self.lock=threading.Lock();self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='letf-lab')

    def prepare(self,body):
        if set(body)-{'profile','request','annual_contribution','annual_roth_magi','annual_eligible_compensation'}:
            raise ValueError('Unknown dashboard inputs')
        profile=body['profile']
        if profile not in self.profiles:raise ValueError('Unknown registered profile')
        request=copy.deepcopy(body['request']);original=self.profiles[profile]
        for key in ('bundles','signals'):
            if request.get(key)!=original.get(key):raise ValueError('Browser cannot change registered private source paths')
        annual=body.get('annual_contribution')
        if annual is not None:
            if isinstance(annual,bool) or not np.isfinite(annual) or annual<0:raise ValueError('Contribution must be finite/nonnegative')
            if annual and 'roth' in request.get('accounts',['taxable','roth']):
                for key in ('annual_roth_magi','annual_eligible_compensation'):
                    value=body.get(key)
                    if value is None or isinstance(value,bool) or not np.isfinite(value) or value<0:
                        raise ValueError('Supply separate Roth MAGI and eligible compensation; ordinary income is insufficient')
                if not request.get('roth_owner'):raise ValueError('Roth owner history required before contribution generation')
            market,*_=load_portfolio(request)
            dates=market.loc[request['start']:request['end']].index
            flows=[f for f in request.get('cash_flows',[]) if f['kind']=='withdrawal']
            if annual:
                for year in sorted(set(dates.year)):
                    first=dates[dates.year==year][0]
                    flows.append({'date':str(first.date()),'kind':'contribution','amount':annual,
                        'magi':body.get('annual_roth_magi'),'compensation':body.get('annual_eligible_compensation')})
            request['cash_flows']=sorted(flows,key=lambda f:f['date'])
        validate_request(request)
        return request

    def submit(self,body):
        request=self.prepare(body)
        with self.lock:
            if any(j['status']in ('QUEUED','RUNNING') for j in self.jobs.values()):
                raise ValueError('One canonical replay at a time; cancel or finish the active job')
            if len(self.jobs)>=100:raise ValueError('Session job limit reached; restart with retained exports')
            key=uuid.uuid4().hex
            self.jobs[key]={'id':key,'status':'QUEUED','cancel':threading.Event(),'request':request}
        self.executor.submit(self._run,key)
        return {'id':key,'status':'QUEUED'}

    def _run(self,key):
        target=self.root/key
        try:
            with self.lock:self.jobs[key]['status']='RUNNING'
            def progress(value):
                with self.lock:
                    if self.jobs[key]['cancel'].is_set():raise CancelledError('Cancelled before next account event')
                    self.jobs[key]['progress']=value
            with self.lock:request=copy.deepcopy(self.jobs[key]['request'])
            run_observed(request,output=target,progress=progress)
            render_report(target/'report.json',target/'report.html')
            view=report_view(target/'report.json')
            with self.lock:self.jobs[key].update(status='DONE',view=view)
        except CancelledError as error:
            target.mkdir(parents=True,exist_ok=True,mode=0o700)
            private_json(target/'cancelled.json',{'classification':'BLOCKED','status':'CANCELLED','reason':str(error),'completed_performance_result':False})
            with self.lock:self.jobs[key].update(status='CANCELLED',reason=str(error))
        except Exception as error:
            target.mkdir(parents=True,exist_ok=True,mode=0o700)
            private_json(target/'blocked.json',{'classification':'BLOCKED','reason':str(error),'completed_performance_result':False})
            with self.lock:self.jobs[key].update(status='BLOCKED' if isinstance(error,(ValueError,KeyError,OSError)) else 'FAILED',reason=str(error))

    def get(self,key):
        with self.lock:return {k:copy.deepcopy(v) for k,v in self.jobs[key].items() if k not in ('cancel','request')}

    def cancel(self,key):
        with self.lock:
            job=self.jobs[key]
            if job['status']not in ('RUNNING','QUEUED'):raise ValueError('Only active jobs can be cancelled')
            job['cancel'].set()
        return {'status':'CANCELLATION_REQUESTED'}

    def close(self):
        with self.lock:
            for job in self.jobs.values():job['cancel'].set()
        self.executor.shutdown(wait=True,cancel_futures=True)


class LabServer(ThreadingHTTPServer):
    daemon_threads=True

    def __init__(self,profiles,root,port=8765):
        self.token=secrets.token_urlsafe(32);self.manager=Jobs(profiles,root)
        super().__init__(('127.0.0.1',port),Handler)

    def server_close(self):
        super().server_close();self.manager.close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass  # never log private bodies or local CSRF token

    def send(self,value,status=200,html=False):
        raw=value.encode() if html else json.dumps(value,allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json')
        self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"frame-ancestors 'none'")
        self.end_headers();self.wfile.write(raw)

    def allowed(self,api=False):
        port=self.server.server_address[1]
        hosts={f'127.0.0.1:{port}',f'localhost:{port}'}
        if self.headers.get('Host')not in hosts:return False
        origin=self.headers.get('Origin')
        if origin is not None and origin not in {'http://'+h for h in hosts}:return False
        return not api or secrets.compare_digest(self.headers.get('X-Lab-Token',''),self.server.token)

    def do_GET(self):
        if not self.allowed(api=self.path.startswith('/api/')):self.send({'reason':'Loopback host/origin/token required'},403);return
        try:
            if self.path=='/':self.send(document(server_token=self.server.token),html=True)
            elif self.path=='/api/profiles':self.send({'profiles':self.server.manager.profiles})
            elif self.path.startswith('/api/jobs/'):
                self.send(self.server.manager.get(self.path.removeprefix('/api/jobs/')))
            else:self.send({'reason':'Unknown local route'},404)
        except KeyError:self.send({'reason':'Unknown job'},404)

    def do_POST(self):
        if not self.allowed(api=True):self.send({'reason':'Loopback host/origin/token required'},403);return
        try:
            if self.headers.get_content_type()!='application/json':raise ValueError('JSON requests only')
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=1024*1024:raise ValueError('Bounded JSON request body required')
            body=json.loads(self.rfile.read(length))
            if self.path=='/api/run':result=self.server.manager.submit(body)
            elif self.path=='/api/prepare':result={'request':self.server.manager.prepare(body)}
            elif self.path.startswith('/api/jobs/') and self.path.endswith('/cancel'):
                result=self.server.manager.cancel(self.path[len('/api/jobs/'):-len('/cancel')])
            else:self.send({'reason':'Unknown local route'},404);return
            self.send(result)
        except (ValueError,KeyError,TypeError,OSError) as error:self.send({'classification':'BLOCKED','reason':str(error)},400)


def serve(profiles,root,*,port=8765):
    server=LabServer(profiles,root,port)
    print(f'Local laboratory: http://127.0.0.1:{server.server_address[1]} (no external listener)',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
