from AlgorithmImports import *
from collections import defaultdict, deque
from datetime import datetime
import csv
import io
import math
import statistics


class BottleneckWinnerFreePIT(QCAlgorithm):
    """
    QuantConnect cloud backtest for the FREE_DISCOVERY Bottleneck Winner tournament.

    Design principles:
    - point-in-time tradable universe and security identity from QuantConnect
    - includes delisted securities through QuantConnect's PIT universe/security master
    - accounting factor values come from SEC as-filed formation-date features
    - joins SEC fundamentals and SEC competition metrics using point-in-time CIK
    - no current-constituent universe
    - no use of future financial statement values
    - frozen model weights from research/free_stage1/frozen_spec.json

    Required data parameters (choose URL or Object Store key for each source):
        fundamentals_url=<SEC formation-date CSV URL>
        fundamentals_object_key=<QuantConnect Object Store key>
            one of the two is required for every model except pure momentum
        network_url=<CIK network CSV URL>
        network_object_key=<QuantConnect Object Store key>
            one of the two is required for bottleneck models

    Optional parameters:
        model=momentum|quality_momentum|bottleneck_core|bottleneck_momentum|bottleneck_full
        top_n=10|20|40
        start=2009-01-01
        end=2023-12-31
        min_cross_section=20  # frozen full-run default; may be 10 for tiny smoke bundles only
    """

    MODEL_WEIGHTS = {
        "momentum": {
            "momentum12": 0.70,
            "high52": 0.30,
        },
        "quality_momentum": {
            "momentum12": 0.35,
            "high52": 0.15,
            "quality": 0.30,
            "dilution": 0.20,
        },
        "bottleneck_core": {
            "demand_accel": 0.35,
            "pricing_power": 0.30,
            "competitive_scarcity": 0.35,
        },
        "bottleneck_momentum": {
            "demand_accel": 0.20,
            "pricing_power": 0.15,
            "competitive_scarcity": 0.20,
            "momentum12": 0.30,
            "high52": 0.15,
        },
        "bottleneck_full": {
            "demand_accel": 0.15,
            "pricing_power": 0.10,
            "competitive_scarcity": 0.15,
            "quality": 0.15,
            "dilution": 0.10,
            "valuation": 0.10,
            "momentum12": 0.15,
            "high52": 0.10,
        },
    }

    def initialize(self):
        start = self.get_parameter("start") or "2009-01-01"
        end = self.get_parameter("end") or "2023-12-31"
        self.set_start_date(*[int(x) for x in start.split("-")])
        self.set_end_date(*[int(x) for x in end.split("-")])
        self.set_cash(10_000_000)

        self.model_name = self.get_parameter("model") or "bottleneck_full"
        if self.model_name not in self.MODEL_WEIGHTS:
            raise ValueError(f"unknown model={self.model_name}")
        self.top_n = int(self.get_parameter("top_n") or "20")
        if self.top_n not in (10, 20, 40):
            raise ValueError("top_n must be 10, 20, or 40")
        self.min_cross_section = int(self.get_parameter("min_cross_section") or "20")
        if self.min_cross_section < 10:
            raise ValueError("min_cross_section must be >= 10")

        self.sec_fundamentals = self._load_sec_fundamentals(
            self.get_parameter("fundamentals_url"),
            self.get_parameter("fundamentals_object_key"),
        )
        self.network = self._load_network(
            self.get_parameter("network_url"),
            self.get_parameter("network_object_key"),
        )
        self.price_history = defaultdict(lambda: deque(maxlen=270))
        self.audit_rows = []
        self.one_way_cost_bps = 15.0

        # Frozen tournament cost convention: 15 bps all-in execution drag on
        # every buy/sell side, with no separate commission. Add (rather than
        # replace) an initializer so LEAN's other brokerage reality models
        # remain intact.
        self.add_security_initializer(self._initialize_security_costs)
        self._last_qkey = None
        self._pending_targets = []
        self._latest_scores = {}
        self._latest_feature_rows = {}
        self._selection_date = None

        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.set_benchmark(self.spy)

        self.universe_settings.resolution = Resolution.DAILY
        self.universe_settings.data_normalization_mode = DataNormalizationMode.ADJUSTED
        self.add_universe(self._select_fundamentals)

        # Prime one trading year of adjusted-price history before 2009-01-01.
        # Universe selection runs during warm-up, so _select_fundamentals can
        # populate price_history without placing trades.
        self.set_warm_up(252, Resolution.DAILY)

        # Rebalance after universe selection at the market open.
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open(self.spy, 1),
            self._rebalance_if_pending,
        )

        self.debug(
            f"FREE_DISCOVERY model={self.model_name} top_n={self.top_n} "
            f"min_cross_section={self.min_cross_section} "
            f"sec_fund_rows={sum(len(v) for v in self.sec_fundamentals.values())} "
            f"network_rows={sum(len(v) for v in self.network.values())}"
        )

    def _initialize_security_costs(self, security):
        if security.type == SecurityType.EQUITY:
            security.set_fee_model(ConstantFeeModel(0, "USD"))
            security.set_slippage_model(ConstantSlippageModel(self.one_way_cost_bps / 10_000.0))

    # ---------- External CSV sources ----------

    def _load_csv_text(self, url, object_key, label):
        if object_key:
            if not self.object_store.contains_key(object_key):
                raise ValueError(f"{label} Object Store key not found: {object_key}")
            raw = self.object_store.read(object_key)
            if not raw:
                raise ValueError(f"{label} Object Store key is empty: {object_key}")
            return raw
        if url:
            raw = self.download(url)
            if not raw:
                raise ValueError(f"{label} URL returned empty content")
            return raw
        return ""

    # ---------- SEC as-filed fundamentals ----------

    def _load_sec_fundamentals(self, url, object_key):
        # cik -> sorted list of (formation_date, metrics)
        out = defaultdict(list)
        raw = self._load_csv_text(url, object_key, "fundamentals")
        if not raw:
            if self.model_name != "momentum":
                self.debug(
                    "No fundamentals_url or fundamentals_object_key supplied; "
                    "SEC accounting features will be missing."
                )
            return out
        for row in csv.DictReader(io.StringIO(raw)):
            cik = str(row["cik"]).zfill(10)
            formation_date = datetime.strptime(
                row["formation_date"][:10], "%Y-%m-%d"
            ).date()
            information_date = datetime.strptime(
                row["information_date"][:10], "%Y-%m-%d"
            ).date()
            if information_date > formation_date:
                raise ValueError(
                    f"future SEC filing in fundamentals_url: cik={cik} "
                    f"information_date={information_date} formation_date={formation_date}"
                )
            out[cik].append(
                (
                    formation_date,
                    {
                        "information_date": information_date,
                        "revenue": self._float(row.get("revenue")),
                        "revenue_qtrs": self._float(row.get("revenue_qtrs")),
                        "gross_margin": self._float(row.get("gross_margin")),
                        "operating_margin": self._float(row.get("operating_margin")),
                        "fcf_margin": self._float(row.get("fcf_margin")),
                        "leverage": self._float(row.get("leverage")),
                        "revenue_growth_yoy": self._float(row.get("revenue_growth_yoy")),
                        "gross_margin_change_yoy": self._float(
                            row.get("gross_margin_change_yoy")
                        ),
                        "operating_margin_change_yoy": self._float(
                            row.get("operating_margin_change_yoy")
                        ),
                        "share_growth_yoy": self._float(row.get("share_growth_yoy")),
                    },
                )
            )
        for cik in out:
            out[cik].sort(key=lambda x: x[0])
        return out

    def _sec_fundamental_asof(self, cik, d):
        rows = self.sec_fundamentals.get(str(cik).zfill(10), ())
        best = None
        for formation_date, metrics in rows:
            if formation_date <= d:
                if metrics["information_date"] > d:
                    raise ValueError(
                        f"SEC PIT invariant violated for cik={cik} at {d}"
                    )
                best = metrics
            else:
                break
        return best

    # ---------- Network ----------

    def _load_network(self, url, object_key):
        # cik -> sorted list of (formation_date, metrics)
        out = defaultdict(list)
        raw = self._load_csv_text(url, object_key, "network")
        if not raw:
            if self.model_name.startswith("bottleneck"):
                self.debug(
                    "No network_url or network_object_key supplied; "
                    "bottleneck features will be missing."
                )
            return out
        for row in csv.DictReader(io.StringIO(raw)):
            cik = str(row["cik"]).zfill(10)
            d = datetime.strptime(row["formation_date"][:10], "%Y-%m-%d").date()
            out[cik].append(
                (
                    d,
                    {
                        "peer_count": self._float(row.get("peer_count")),
                        "total_similarity": self._float(row.get("total_similarity")),
                        "similarity_hhi": self._float(row.get("similarity_hhi")),
                        "text_scarcity_raw": self._float(row.get("text_scarcity_raw")),
                    },
                )
            )
        for cik in out:
            out[cik].sort(key=lambda x: x[0])
        return out

    def _network_asof(self, cik, d):
        rows = self.network.get(str(cik).zfill(10), ())
        best = None
        for dt, metrics in rows:
            if dt <= d:
                best = metrics
            else:
                break
        return best

    # ---------- Fundamental state ----------

    @staticmethod
    def _float(x):
        try:
            v = float(x)
            return v if math.isfinite(v) else float("nan")
        except Exception:
            return float("nan")

    @staticmethod
    def _safe_div(a, b):
        try:
            a = float(a)
            b = float(b)
            if not math.isfinite(a) or not math.isfinite(b) or b == 0:
                return float("nan")
            return a / b
        except Exception:
            return float("nan")

    @staticmethod
    def _eligible(f):
        try:
            if not f.has_fundamental_data or f.adjusted_price < 5 or f.dollar_volume < 1_000_000:
                return False
            sr = f.security_reference
            if sr.security_type != "ST00000001" or not sr.is_primary_share:
                return False
            if sr.exchange_id not in ("NYS", "NAS", "ASE"):
                return False
            # No sector/SIC exclusions are part of the frozen Stage-1 universe.
            # Names with missing factor inputs are removed later by the scoring
            # completeness rule, not by an undocumented industry filter.
            return True
        except Exception:
            return False

    def _update_price(self, f):
        p = self._float(f.adjusted_price)
        if math.isfinite(p) and p > 0:
            self.price_history[f.symbol].append(p)

    def _is_rebalance_window(self):
        # First trading days of Jan/Apr/Jul/Oct. All information used here is
        # available on or before current algorithm date.
        if self.time.month not in (1, 4, 7, 10):
            return False
        qkey = (self.time.year, self.time.month)
        return qkey != self._last_qkey

    def _feature_row(self, f):
        # Frozen 12-month momentum and 52-week-high measures require a full
        # trading-year history. Do not substitute a shorter early-sample window.
        px = list(self.price_history[f.symbol])
        if len(px) < 252:
            return None
        p_now = px[-1]
        p_12m = px[-252]
        momentum12 = self._safe_div(p_now, p_12m)
        if math.isfinite(momentum12):
            momentum12 -= 1
        high52 = self._safe_div(p_now, max(px[-252:]))

        cik = None
        try:
            cik = str(f.company_reference.cik).strip().zfill(10)
        except Exception:
            pass

        sec = self._sec_fundamental_asof(cik, self.time.date()) if cik else None
        if sec:
            demand_accel = sec["revenue_growth_yoy"]
            pricing_power = (
                sec["gross_margin_change_yoy"]
                if math.isfinite(sec["gross_margin_change_yoy"])
                else sec["operating_margin_change_yoy"]
            )
            quality = (
                sec["fcf_margin"] - sec["leverage"]
                if math.isfinite(sec["fcf_margin"]) and math.isfinite(sec["leverage"])
                else (
                    sec["operating_margin"] - sec["leverage"]
                    if math.isfinite(sec["operating_margin"])
                    and math.isfinite(sec["leverage"])
                    else float("nan")
                )
            )
            dilution = (
                -sec["share_growth_yoy"]
                if math.isfinite(sec["share_growth_yoy"])
                else float("nan")
            )

            # Put 10-K and 10-Q revenue on the same scale before using it
            # in valuation. SEC qtrs=4 is annual; qtrs=1 is one quarter, so
            # annualize the latter by 4. This is PIT and internally comparable,
            # though still intentionally simpler than a full TTM reconstruction.
            market_cap = self._float(f.market_cap)
            revenue_qtrs = sec["revenue_qtrs"]
            annualized_revenue = float("nan")
            if math.isfinite(sec["revenue"]) and math.isfinite(revenue_qtrs):
                if int(revenue_qtrs) == 4:
                    annualized_revenue = sec["revenue"]
                elif int(revenue_qtrs) == 1:
                    annualized_revenue = sec["revenue"] * 4.0
            sales_multiple = self._safe_div(market_cap, annualized_revenue)
            valuation = (
                -math.log1p(max(sales_multiple, 0))
                if math.isfinite(sales_multiple)
                else float("nan")
            )
            fund_file_date = sec["information_date"]
        else:
            market_cap = self._float(f.market_cap)
            demand_accel = float("nan")
            pricing_power = float("nan")
            quality = float("nan")
            dilution = float("nan")
            valuation = float("nan")
            fund_file_date = None

        net = self._network_asof(cik, self.time.date()) if cik else None
        scarcity = net["text_scarcity_raw"] if net else float("nan")

        return {
            "symbol": f.symbol,
            "cik": cik,
            "market_cap": market_cap,
            # Frozen label retained for the tournament: this Stage-1 field is
            # the SEC as-filed YoY revenue-growth signal.
            "demand_accel": demand_accel,
            "pricing_power": pricing_power,
            "competitive_scarcity": scarcity,
            "quality": quality,
            "dilution": dilution,
            "valuation": valuation,
            "momentum12": momentum12,
            "high52": high52,
            "fund_file_date": fund_file_date,
        }

    # ---------- Ranking ----------

    def _winsor_z(self, rows, feature):
        vals = [(i, r[feature]) for i, r in enumerate(rows) if math.isfinite(r.get(feature, float("nan")))]
        if len(vals) < self.min_cross_section:
            return {}
        raw = sorted(v for _, v in vals)
        lo = raw[max(0, int(0.01 * (len(raw) - 1)))]
        hi = raw[min(len(raw) - 1, int(0.99 * (len(raw) - 1)))]
        clipped = [(i, min(max(v, lo), hi)) for i, v in vals]
        xs = [v for _, v in clipped]
        mu = statistics.fmean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0
        if sd <= 0:
            return {}
        return {i: (v - mu) / sd for i, v in clipped}

    def _score(self, rows):
        weights = self.MODEL_WEIGHTS[self.model_name]
        z = {feature: self._winsor_z(rows, feature) for feature in weights}
        scored = []
        for i, row in enumerate(rows):
            parts = []
            missing = False
            for feature, w in weights.items():
                if i not in z[feature]:
                    missing = True
                    break
                parts.append(w * z[feature][i])
            if missing:
                continue
            score = sum(parts)
            scored.append((score, row))
        return sorted(scored, key=lambda x: x[0], reverse=True)

    # ---------- Universe / trading ----------

    def _select_fundamentals(self, fundamentals):
        eligible = []
        for f in fundamentals:
            # Price history must be calendar/trading-day history, not "days the
            # stock happened to pass today's liquidity screen".
            self._update_price(f)
            if self._eligible(f):
                eligible.append(f)

        # Warm-up is used only to prime trailing price history. Universe
        # selection still streams fundamentals during warm-up, but no portfolio
        # state or rebalance key should advance before the official start date.
        if self.is_warming_up:
            return Universe.UNCHANGED

        if not self._is_rebalance_window():
            return Universe.UNCHANGED

        self._last_qkey = (self.time.year, self.time.month)
        rows = []
        for f in eligible:
            row = self._feature_row(f)
            if row:
                rows.append(row)

        scored = self._score(rows)
        selected = scored[: self.top_n]

        for rank, (score, row) in enumerate(selected, 1):
            audit = {
                "formation_date": str(self.time.date()),
                "model": self.model_name,
                "top_n": self.top_n,
                "eligible_count": len(eligible),
                "feature_complete_count": len(rows),
                "scored_count": len(scored),
                "rank": rank,
                "score": score,
                "symbol": str(row["symbol"]),
                "cik": row.get("cik"),
            }
            for feature in self.MODEL_WEIGHTS[self.model_name]:
                audit[feature] = row.get(feature)
            audit["market_cap"] = row.get("market_cap")
            audit["fund_file_date"] = row.get("fund_file_date")
            self.audit_rows.append(audit)

        self._pending_targets = [r["symbol"] for _, r in selected]
        self._latest_scores = {r["symbol"]: s for s, r in selected}
        self._latest_feature_rows = {r["symbol"]: r for _, r in selected}
        self._selection_date = self.time.date()

        if not self._pending_targets:
            self.debug(f"{self.time.date()} no complete scored names")
            return Universe.UNCHANGED

        # Hold only selected securities to keep cloud memory manageable.
        return self._pending_targets

    def _rebalance_if_pending(self):
        if not self._pending_targets or self._selection_date != self.time.date():
            return

        targets = set(self._pending_targets)
        for kvp in self.portfolio:
            symbol = kvp.key
            holding = kvp.value
            if holding.invested and symbol not in targets and symbol != self.spy:
                self.liquidate(symbol, tag="quarterly exit")

        weight = 1.0 / len(self._pending_targets)
        for symbol in self._pending_targets:
            self.set_holdings(symbol, weight, tag=f"{self.model_name}:{self.top_n}")

        self.debug(
            f"{self.time.date()} rebalance model={self.model_name} n={len(self._pending_targets)}"
        )
        self._pending_targets = []

    def on_end_of_algorithm(self):
        key = (
            f"{self.project_id}/free_stage1/"
            f"bottleneck_{self.model_name}_top{self.top_n}_{self.algorithm_id}.csv"
        )
        saved = False
        if self.audit_rows:
            fields = list(self.audit_rows[0].keys())
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.audit_rows)
            saved = self.object_store.save(key, buf.getvalue())
        self.log(
            f"EVIDENCE_LABEL=FREE_DISCOVERY model={self.model_name} top_n={self.top_n} "
            f"one_way_cost_bps={self.one_way_cost_bps:.1f} "
            f"audit_rows={len(self.audit_rows)} audit_saved={saved} audit_key={key}"
        )
