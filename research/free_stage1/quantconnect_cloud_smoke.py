#!/usr/bin/env python3
"""QuantConnect Stage-1 integration smoke runner.

This intentionally avoids reading or emitting performance statistics. It validates
only compilation, runtime/integration behavior, order mechanics, and audit saves.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://www.quantconnect.com/api/v2"
TERMINAL_STATES = {"BuildSuccess", "BuildError"}


class QCError(RuntimeError):
    pass


class QCClient:
    def __init__(self, user_id: str, api_token: str, organization_id: str):
        self.user_id = str(user_id)
        self.api_token = api_token
        self.organization_id = organization_id
        self.session = requests.Session()

    def _headers(self):
        ts = str(int(time.time()))
        digest = hashlib.sha256(f"{self.api_token}:{ts}".encode()).hexdigest()
        auth = base64.b64encode(f"{self.user_id}:{digest}".encode()).decode()
        return {"Authorization": f"Basic {auth}", "Timestamp": ts}

    def post(self, endpoint, *, payload=None, data=None, files=None):
        response = self.session.post(
            f"{BASE_URL}/{endpoint.lstrip('/')}",
            headers=self._headers(),
            json=payload,
            data=data,
            files=files,
            timeout=90,
        )
        response.raise_for_status()
        out = response.json()
        if not out.get("success", False):
            raise QCError(f"{endpoint} failed: {out.get('errors') or out}")
        return out

    def authenticate(self):
        self.post("authenticate", payload={})

    def get_or_create_project(self, name: str) -> int:
        out = self.post("projects/read", payload={"start": 0, "end": 1000})
        matches = [
            p for p in out.get("projects", [])
            if p.get("name") == name
            and (not self.organization_id or p.get("organizationId") == self.organization_id)
        ]
        if len(matches) > 1:
            raise QCError(f"multiple QuantConnect projects named {name!r}")
        if matches:
            return int(matches[0]["projectId"])

        payload = {"name": name, "language": "Py"}
        if self.organization_id:
            payload["organizationId"] = self.organization_id
        out = self.post("projects/create", payload=payload)
        projects = out.get("projects", [])
        if not projects:
            raise QCError(f"project creation returned no project for {name!r}")
        return int(projects[0]["projectId"])

    def sync_main(self, project_id: int, source_path: Path):
        source = source_path.read_text(encoding="utf-8")
        out = self.post(
            "files/read",
            payload={"projectId": project_id, "includeLibraries": False},
        )
        names = {f.get("name") for f in out.get("files", [])}
        payload = {
            "projectId": project_id,
            "name": "main.py",
            "content": source,
            "codeSourceId": "GitHub Stage1 pre-return smoke",
        }
        if "main.py" in names:
            self.post("files/update", payload=payload)
        else:
            payload.pop("codeSourceId", None)
            self.post("files/create", payload=payload)

    def upload_object(self, key: str, source_path: Path):
        if not self.organization_id:
            raise QCError("QC_ORGANIZATION_ID is required for Object Store uploads")
        with source_path.open("rb") as fh:
            self.post(
                "object/set",
                data={"organizationId": self.organization_id, "key": key},
                files={"objectData": (source_path.name, fh, "text/csv")},
            )

    def compile(self, project_id: int, timeout_s: int = 300) -> str:
        out = self.post("compile/create", payload={"projectId": project_id})
        compile_id = out["compileId"]
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            out = self.post(
                "compile/read",
                payload={"projectId": project_id, "compileId": compile_id},
            )
            state = out.get("state")
            if state in TERMINAL_STATES:
                if state != "BuildSuccess":
                    raise QCError("compile failed:\n" + "\n".join(out.get("logs", [])))
                return compile_id
            time.sleep(3)
        raise QCError(f"compile timed out for project {project_id}")

    def run_backtest(
        self,
        project_id: int,
        compile_id: str,
        name: str,
        parameters: dict,
        timeout_s: int = 900,
    ):
        out = self.post(
            "backtests/create",
            payload={
                "projectId": project_id,
                "compileId": compile_id,
                "backtestName": name,
                "parameters": parameters,
            },
        )
        backtest_id = out["backtest"]["backtestId"]
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            out = self.post(
                "backtests/read",
                payload={"projectId": project_id, "backtestId": backtest_id},
            )
            bt = out.get("backtest", {})
            if bt.get("completed") or bt.get("status") == "Runtime Error":
                if bt.get("status") == "Runtime Error" or bt.get("hasInitializeError"):
                    raise QCError(
                        f"backtest runtime failure {backtest_id}: "
                        f"{bt.get('error')}\n{bt.get('stacktrace')}"
                    )
                return backtest_id
            time.sleep(5)
        raise QCError(f"backtest timed out: {backtest_id}")

    def read_logs(self, project_id: int, backtest_id: str):
        first = self.post(
            "backtests/read/log",
            payload={
                "projectId": project_id,
                "backtestId": backtest_id,
                "start": 0,
                "end": 200,
                "query": None,
            },
        )
        logs = list(first.get("logs", []))
        length = int(first.get("length", len(logs)))
        start = 200
        while start < length:
            end = min(start + 200, length)
            page = self.post(
                "backtests/read/log",
                payload={
                    "projectId": project_id,
                    "backtestId": backtest_id,
                    "start": start,
                    "end": end,
                    "query": None,
                },
            )
            logs.extend(page.get("logs", []))
            start = end
        return logs

    def read_orders(self, project_id: int, backtest_id: str, max_orders: int = 1000):
        orders = []
        start = 0
        step = 90
        while start < max_orders:
            out = self.post(
                "backtests/orders/read",
                payload={
                    "projectId": project_id,
                    "backtestId": backtest_id,
                    "start": start,
                    "end": start + step,
                },
            )
            page = list(out.get("orders", []))
            orders.extend(page)
            if len(page) < step:
                break
            start += step
        return orders


def require(cond: bool, message: str):
    if not cond:
        raise QCError(message)


def parse_int(pattern: str, text: str, label: str) -> int:
    m = re.search(pattern, text)
    if not m:
        raise QCError(f"missing {label} in QuantConnect logs")
    return int(m.group(1))


def order_status(order):
    return str(order.get("status", "")).lower()


def order_type(order):
    return str(order.get("type", "")).lower()


def validate_bottleneck(logs, orders, code_text):
    text = "\n".join(logs)
    require("FREE_DISCOVERY model=bottleneck_core" in text, "Bottleneck initialization log missing")
    m = re.search(r"sec_fund_rows=(\d+)\s+network_rows=(\d+)", text)
    require(m is not None, "Bottleneck SEC/network row counts missing")
    sec_rows, network_rows = int(m.group(1)), int(m.group(2))
    require(sec_rows > 0, "Bottleneck loaded zero SEC rows")
    require(network_rows > 0, "Bottleneck loaded zero network rows")
    require("rebalance model=bottleneck_core n=" in text, "Bottleneck never reached a real rebalance")
    audit_rows = parse_int(r"audit_rows=(\d+)", text, "Bottleneck audit row count")
    require(audit_rows >= 10, f"Bottleneck audit only has {audit_rows} rows")
    require("audit_saved=True" in text or "audit_saved=true" in text, "Bottleneck audit was not saved")
    require("self.one_way_cost_bps = 15.0" in code_text, "frozen 15 bps cost setting missing from executed source")
    require(len(orders) > 0, "Bottleneck produced no orders")
    invalid = [o for o in orders if order_status(o) in {"7", "invalid"}]
    require(not invalid, f"Bottleneck has {len(invalid)} invalid orders")
    return {
        "sec_rows": sec_rows,
        "network_rows": network_rows,
        "audit_rows": audit_rows,
        "order_count": len(orders),
    }


def validate_leaps(logs, orders):
    text = "\n".join(logs)
    require("EVIDENCE_LABEL=FREE_DISCOVERY underlying=SPY" in text, "LEAPS initialization log missing")
    require(" QUEUE " in text, "LEAPS never selected/queued a real historical option")
    entries = parse_int(r"LEAPS_AUDIT entries=(\d+)", text, "LEAPS entry count")
    rolls = parse_int(r"rolls=(\d+)", text, "LEAPS roll count")
    selection_rows = parse_int(r"selection_rows=(\d+)", text, "LEAPS selection audit count")
    fill_rows = parse_int(r"fill_rows=(\d+)", text, "LEAPS fill audit count")
    require(entries >= 1, "LEAPS produced no filled entry")
    require(rolls >= 1, "LEAPS smoke did not exercise roll sequencing")
    require(selection_rows >= 2, "LEAPS smoke did not record entry + replacement selections")
    require(fill_rows >= 3, "LEAPS smoke has too few fill-audit rows to cover entry/exit/re-entry")
    require(
        "selection_saved=True" in text or "selection_saved=true" in text,
        "LEAPS selection audit was not saved",
    )
    require(
        "fill_saved=True" in text or "fill_saved=true" in text,
        "LEAPS fill audit was not saved",
    )
    require(len(orders) > 0, "LEAPS produced no orders")
    invalid = [o for o in orders if order_status(o) in {"7", "invalid"}]
    require(not invalid, f"LEAPS has {len(invalid)} invalid orders")
    moo = [o for o in orders if order_type(o) in {"4", "marketonopen", "market on open"}]
    require(moo, "LEAPS order history contains no Market-On-Open orders")
    return {
        "entries": entries,
        "rolls": rolls,
        "selection_rows": selection_rows,
        "fill_rows": fill_rows,
        "order_count": len(orders),
        "moo_order_count": len(moo),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec-features", required=True, type=Path)
    ap.add_argument("--network", required=True, type=Path)
    ap.add_argument("--bottleneck-code", required=True, type=Path)
    ap.add_argument("--leaps-code", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--source-sha", default=os.getenv("GITHUB_SHA", "unknown"))
    args = ap.parse_args()

    for path in (args.sec_features, args.network, args.bottleneck_code, args.leaps_code):
        require(path.exists(), f"missing required file: {path}")

    required_env = ["QC_USER_ID", "QC_API_TOKEN", "QC_ORGANIZATION_ID"]
    missing = [k for k in required_env if not os.getenv(k)]
    require(not missing, "missing QuantConnect credentials/config: " + ", ".join(missing))

    qc = QCClient(
        os.environ["QC_USER_ID"],
        os.environ["QC_API_TOKEN"],
        os.environ["QC_ORGANIZATION_ID"],
    )
    qc.authenticate()

    suffix = args.source_sha[:12]
    fund_key = f"free_stage1/smoke/{suffix}/sec_formation_features.csv"
    network_key = f"free_stage1/smoke/{suffix}/network_metrics_all.csv"
    qc.upload_object(fund_key, args.sec_features)
    qc.upload_object(network_key, args.network)

    report = {
        "source_sha": args.source_sha,
        "evidence_label": "INTEGRATION_SMOKE_ONLY",
        "performance_statistics_inspected_or_emitted": False,
        "object_store": {"fundamentals_key": fund_key, "network_key": network_key},
        "tests": {},
    }

    bottleneck_project = qc.get_or_create_project("LETF Stage1 Smoke/BottleneckWinnerFreePIT")
    qc.sync_main(bottleneck_project, args.bottleneck_code)
    bottleneck_compile = qc.compile(bottleneck_project)
    bottleneck_id = qc.run_backtest(
        bottleneck_project,
        bottleneck_compile,
        f"PRE-RETURN Bottleneck smoke {suffix}",
        {
            "model": "bottleneck_core",
            "top_n": 10,
            "min_cross_section": 10,
            "start": "2024-01-02",
            "end": "2024-03-29",
            "fundamentals_object_key": fund_key,
            "network_object_key": network_key,
        },
    )
    bottleneck_logs = qc.read_logs(bottleneck_project, bottleneck_id)
    bottleneck_orders = qc.read_orders(bottleneck_project, bottleneck_id)
    report["tests"]["bottleneck"] = {
        "project_id": bottleneck_project,
        "backtest_id": bottleneck_id,
        "checks": validate_bottleneck(
            bottleneck_logs,
            bottleneck_orders,
            args.bottleneck_code.read_text(encoding="utf-8"),
        ),
    }

    leaps_project = qc.get_or_create_project("LETF Stage1 Smoke/RealHistoricalLeaps")
    qc.sync_main(leaps_project, args.leaps_code)
    leaps_compile = qc.compile(leaps_project)
    leaps_id = qc.run_backtest(
        leaps_project,
        leaps_compile,
        f"PRE-RETURN LEAPS smoke {suffix}",
        {
            "underlying": "SPY",
            "target_delta": 0.80,
            "target_dte": 548,
            "allocation": 1.00,
            "slippage_bps": 5,
            "min_oi": 100,
            "start": "2023-01-03",
            "end": "2024-02-01",
        },
    )
    leaps_logs = qc.read_logs(leaps_project, leaps_id)
    leaps_orders = qc.read_orders(leaps_project, leaps_id)
    report["tests"]["leaps"] = {
        "project_id": leaps_project,
        "backtest_id": leaps_id,
        "checks": validate_leaps(leaps_logs, leaps_orders),
    }

    report["go_no_go"] = "GO_FOR_177_MATRIX"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QC STAGE1 SMOKE FAIL: {exc}", file=sys.stderr)
        raise
