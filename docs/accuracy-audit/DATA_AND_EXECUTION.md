# Data, reconstruction and execution scope

Audit retrieval date: 2026-11-23. Machine-readable provenance is in `evidence/public-input-manifest.json`. Local raw snapshots are in `audit-output/public-daily-inputs-network/`; they are not redistributed by Git. Preserve them to reproduce the exact results, since future vendor downloads may be revised.

## Observation versus reconstruction

`OBSERVED` means a sourced price/return observation, not proof of a realizable fill. `VALIDATED RECONSTRUCTION` means a proxy passed the stated held-out numerical test for that interval and source convention. `HYPOTHETICAL STRESS MODEL` includes resampled paths, approximate pre-inception products, assumed costs and retrospective parameter fits. `UNSUPPORTED` means the inputs/mechanics cannot establish the requested claim.

Public captures use Yahoo Finance via pinned yfinance 0.2.66. Vendor OHLC may be split-restated, and adjusted close incorporates distributions and splits. **Neither is certified as an as-traded historical execution quote.** The account replay requires separately verified as-traded prices, corporate actions and distribution payable dates. The current capture provides useful fund-return validation inputs, not full historical taxable execution evidence. No fund expenses or financing are deducted again from observed adjusted fund returns.

Earliest observations in this capture (not necessarily legal inception):

| Series | First captured observation | Convention |
|---|---|---|
| SPY | 1993-01-29 | Vendor adjusted close total-return proxy |
| QQQ | 1999-03-10 | Same |
| TQQQ | 2010-02-11 | Same |
| QLD / SSO | 2006-06-21 | Same |
| UPRO | 2009-06-25 | Same |
| TLT | 2002-07-30 | Same |
| TMF | 2009-04-16 | Same |
| Federal funds effective rate | 1990-01-01 requested start | Annual rate quoted in percentage points; daily calendar observations |

The rate source is the [Federal Reserve DFF series through FRED](https://fred.stlouisfed.org/series/DFF). It is a funding proxy, not an actual fund swap invoice. The audit lags it two calendar days; the snapshot is a current publication vintage, not a preserved real-time historical vintage. Original data/cache files and `validation_metadata.json` were not rewritten.

## Daily method and frozen test

For a long daily-reset fund: daily return = leverage × benchmark daily return − annual fund fee × elapsed calendar days / 365 − (leverage−1) × (start-of-interval funding rate + spread) × elapsed calendar days / 360. Simulated 252-session years use an explicit 1/252 interval instead. Compounding generates path dependence; no additional volatility-decay fee is applied. The zero-NAV boundary is absorbing under the hypothetical liquidation convention.

Training ends 2019-12-31; evaluation is 2020-01-01 through 2025-12-31. The spread is an intercept-free nonnegative least-squares estimate on training observations only. Fee assumptions are fixed in the script, not tuned on evaluation. The preregistered [gate](reconstruction-gate.json) requires 504 observations, daily RMSE ≤30 bp, absolute daily bias ≤2 bp, annual residual volatility ≤5%, CAGR error ≤2 percentage points, drawdown error ≤5 points and terminal relative error ≤15%. It was written before this audit evaluation. Existing research holdouts were not relabeled or overwritten; this newly defined interval is not claimed to be historically unseen by all previous research.

Results are in [daily-reconstruction-v1.json](evidence/daily-reconstruction-v1.json), with yearly, lagged-volatility-regime and holding-period diagnostics. Regime slices are noncontiguous; only residual statistics are reported for them. Compound holding-period comparisons retain contiguous observations. These are fixed-model forecast diagnostics, not after-tax strategy rankings.

| Product | Absolute CAGR error | Annual residual volatility | Numerical gate |
|---|---:|---:|---|
| TQQQ | 2.303 pp | 3.528% | FAIL |
| QLD | 0.554 pp | 1.781% | PASS |
| UPRO | 2.082 pp | 2.874% | FAIL |
| SSO | 0.537 pp | 1.842% | PASS |
| TMF | 0.546 pp | 3.966% | PASS |

Daily pre-inception proxy files were produced for all five. Their earliest usable benchmark periods begin in 1993 (SPY), 1999 (QQQ) or 2002 (TLT), subject to the initial return/rate warmup. Every pre-inception row remains hypothetical: historical product fee schedules, actual swaps, exact underlying index conventions and availability risk are not established. Parameters fitted through 2019 are **not** presented as contemporaneous pre-2010 trading knowledge. Monthly observations cannot determine a unique daily-reset path and are rejected.

## Benchmark investigations

- **KMLM:** the [issuer FAQ](https://kraneshares.com/kmlm-managed-futures-faq/) describes 22 contracts across commodities, currencies and bonds and discusses index history to 1988; the [2020 issuer presentation](https://kfafunds.com/resources/presentation/2020_12_31_kmlm_presentation.pdf) gives constituents/methodology context. The ETF's 2020 inception must not be backdated to the index's retrospectively published history. Authentic dated constituent prices, roll rules, weights, historical publication vintages and fees were not captured here. No arbitrary managed-futures drift is accepted.
- **VBR:** the [2014 SEC-filed annual report](https://www.sec.gov/Archives/edgar/data/36405/000093247114004906/index_final.htm) documents its historical benchmark splice and ETF history from January 2004. A CRSP small-value backtest is not automatically the historical portfolio/index available at each earlier date. Current provider pages can change benchmark descriptions; freeze the dated filing. No invented VBR prehistory is introduced.
- **Leveraged gold:** [ProShares UGL product terms](https://www.proshares.com/globalassets/proshares/fact-sheet/prosharesfactsheetugl.pdf) target daily leveraged **gold futures index** performance. Levering spot gold with arbitrary positive drift does not replicate futures rolls/collateral or product taxes. Dated gold contracts, index roll methodology and product-specific tax structure are still required.

## Broker and execution conventions

`PortfolioLedger` enforces settled cash, share quantities, commissions, half-spread plus slippage, collateral reservations, explicit fills and product availability. A requested partial fill must specify the fill quantity. Explicit settlement dates can defer cash availability. Default same-day cash settlement in the backtest runner is an execution **scenario**, not an exchange-calendar/T+1 certification. Complete bid/ask depth, rejected orders and exchange holidays are not reconstructed from closing levels.

Generic taxable and conservative Roth profiles are cash accounts with stock/ETF permissions. `brokers.py` provides dated conservative subsets of [IBKR IRA permissions](https://www.interactivebrokers.com/support/faq/content/1136195127) and [Fidelity's IRA agreement](https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/limited-margin-agreement.pdf), retrieved 2026-11-23. They do not establish a particular person's approval. The IBKR FAQ contains a stale general stock-settlement sentence; that sentence is not used as a historical calendar rule. IRA futures require explicit approval, source date and actual house-margin multiplier/schedules. Current documentation is rejected as historical permission evidence.

Futures use dated symbols, integer contracts, ticks/multipliers, explicit rolls, reserved initial/maintenance margin and daily cash settlement. [CME MES specifications](https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.contractSpecs.html) support the $5 multiplier and 0.25-point tick ($1.25). Tests reconcile these calculations. No independent historical settlement tape was acquired. OHLC adverse extremes can bound potential margin breaches; they do not determine breach time or liquidation fill. A gap can leave debt after forced liquidation. Negative futures prices are permitted when on the tick grid.

Long-option Roth mechanics require observed dated bid/ask quotes, timezone-aware timestamps, quote-age and displayed-size checks, integral contracts, premium cash and permissions. LEAPS are dated long options, not a separate pricing exemption. Theoretical quotes are rejected as executions. No short writing, assignment, adjusted-contract deliverable or physical exercise is silently simulated. Covered-call/put/spread collateral formulas are tested, but they do not constitute a complete strategy execution or tax implementation.

## Legitimate option-data paths; no purchase made

Prices checked 2026-11-23, USD, personal-license terms may differ from business use:

| Provider | Verified offered access | Coverage / limits |
|---|---|---|
| [ThetaData](https://www.thetadata.net/pricing) | $40/month Value; $80 Standard; $160 Pro | Published page lists 6/10/14 years respectively; Value has minute intervals, higher tiers quote-level features. Confirm endpoint entitlement before assuming historical NBBO. |
| [Cboe DataShop](https://datashop.cboe.com/option-quote-intervals) | Product/selection-specific quote; samples | Listed interval/EOD history from January 2012; no verified universal flat price. |
| [HistoricalOptionData / DeltaNeutral](https://historicaloptiondata.com/) | Level 1 most recent year $230; full history $1,150 | Published EOD history from 2002; quotes do not resolve intraday assignment or fill sequencing. |
| [Databento](https://databento.com/pricing) | Historical pay-as-you-go exists; exact quote not verified | Pricing page required JavaScript in this audit. Do not repeat an unverified third-party per-GB price. |

The cheapest verified published subscription among those inspected is ThetaData's $40 tier, with the stated limitations; free samples validate parsers only. This is not a claim to have found the global cheapest executable history or obtained any licensed dataset.
