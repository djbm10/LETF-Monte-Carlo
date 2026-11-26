# Account × instrument × strategy support

This matrix describes strongest demonstrated support, not interchangeable
investment rankings. `OBSERVED_MARKET` identifies underlying source observations;
fills remain `CONDITIONAL_STRESS` unless actual execution is independently proven.
No cell is forecast-approved. Earliest dates below are **reviewed account
coverage**, not a statement of product inception or earliest downloadable NAV.

| Instrument/mode | Taxable | Roth | Earliest reviewed period / strongest evidence |
|---|---|---|---|
| TQQQ actual fund | Supported corporate-RIC subset | Internal trading supported |2021-01-04–2025-12-31;1255 daily rows/account, final issuer distributions/splits; OBSERVED_MARKET inputs |
| UPRO actual fund | Same bounded subset | Internal trading supported |2021-01-04–2025-12-31; same dates/capital/reinvestment/costs as TQQQ |
| SSO actual fund | Same bounded subset | Internal trading supported |2021-01-04–2025-12-31; explicit zero distribution placeholders independently corroborated |
| QLD actual fund |2024–25 reviewed subset | Same period |2023 extension BLOCKED: positive vendor cash conflicts with zero/missing issuer records |
| TMF/QQQ actual funds |2024–25 reviewed subsets | Same periods |Existing final issuer bundles; special state adjustments are independently gated |
| SPY actual fund |2025 reviewed subset, including sourced RIC spillover allocation | Same period |2025-01-02–2025-12-31; earlier issuer classification bundle not certified here |
| Other stocks/ETFs | Schema/API exists | Schema/API exists |BLOCKED performance until nominal daily quotes, complete actions/distributions/tax structure and rights are supplied |
| Federal tax |2021–26 supported income/deduction subset | Routine internal tax zero |Actual law by year; no complete household1040/AMT/credits/preferred-share/complex elections claim |
| Massachusetts |2023–26 bounded law fixtures | Internal tax zero |Actual fund distributions additionally require state source review; federal-only fund runs cannot be relabeled MA |
| FL/TX/NV/FEDERAL_ONLY scenarios | Explicit configurations | Same instrument/account restrictions |No-state-income-tax choices do not eliminate federal, residency or special-fund fact requirements |
| Earlier actual tax law / other states | BLOCKED | Roth pre1998 BLOCKED actual history |Require enacted law and contemporaneous prices/actions; a frozen modern-law scenario must be labeled counterfactual |
| Broad/growth/HiTec daily1x/2x/3x counterfactual | Account history BLOCKED | No actual early Roth claim |French genuine inputs begin1926-07-01;1930–2025 model completed; HISTORICAL_COUNTERFACTUAL |
| Literal Nasdaq-100/TQQQ before benchmark/product | BLOCKED | BLOCKED |Alternative dated growth portfolios are proxies; never relabel them Nasdaq/TQQQ |
| Futures ES/MES/NQ/MNQ/bond/gold schema |Qualifying1256 mechanics fixtures; other tax cases gated |Only explicit dated broker/house-margin profile |Dated contract objects, variation margin, integer sizing, rolls as two explicit trades, gaps/debt; actual historical tape/permissions not certified |
| Standard long options/LEAPS |BLOCKED integrated taxable treatment |Explicitly permitted profile; quote/notice mechanics |Actual timestamp/bid/ask/depth required; standard physical exercise post2024-05-28 T+1 subset; adjusted/other delivery calendars blocked |
| Covered calls / cash-secured puts |BLOCKED interacting taxes |SecuredOptionBook with documented strategy approval |Standard100-share American physical subset post2024-05-28; cash/share locks, liability, explicit assignment/nonassignment fixture tests |
| Collars / spreads / unusual deliverables |BLOCKED |BLOCKED certification |Missing combined-book netting, OCC actions, assignment/hedge/permission and tax evidence; scalar collateral helpers are not complete strategies |
| Cboe free EOD sample |No historical performance claim |Same |2023-08-25 demonstration,32672 rows/34fields; SOFTWARE_FIXTURE, not proof of a complete executable tape |
| KMLM/VBR/gold/sector histories |BLOCKED historical ranking |Same |Need reviewed index/constituent/contract/source rights and distributions; no arbitrary drift substitutes |
| Future5/10/20/30/40-year accounts |Full canonical scenario mechanics |Same market paths; explicit eligibility/history |CONDITIONAL_STRESS, frozen-law assumptions, no accepted model probabilities or full-account precision pass |
| Extinct zero-residual taxable fund |BLOCKED |Explicit sourced internal closure supported |No automatic165(g) deduction from zero NAV; original60 affected cases remain blocked |

## Preserved strategy identities

`legacy_vector_v0` is the original archived wealth implementation, not a
production account ledger. `legacy_rules_v1` preserves the existing causal
cash adapters and rejects unsupported allocations. `causal_cash_v2` explicitly
uses prior-close TR signals/cash caps; original configuration/source hashes are
returned by `python -m letf.lab strategies`. S16 same-bar legacy VIX remains
blocked. Corrected versions do not silently replace historical definitions.

The actual2025 cohort used the same five source bundles, capital, tax profile,
session dates and an **explicit common daily rebalance override**. It is not a
rerun of the original S6/HFEA monthly execution policy. Required long signal
warmup can keep a strategy in cash; no missing prior-year distribution history
was invented. No sorted ranking is released.

| IDs/version in that cohort | Result |
|---|---|
| S1–S19 / causal_cash_v2 |19 cells complete; canonical cash/settlement/tax; sourced VIX for S8/S16 |
| S8,S16 / legacy/current-close variants |Original definitions retained; S16 same-bar variant remains BLOCKED; v2 uses prior closes |
| HFEA / cash_v1 |Complete in common2025 daily override and separate2024–25 default21-session cohort; verified RIC Section199A46-day/91-day tracking |
|35_0 / frozen_ETF_wrapper_v1 |Complete using reviewed QQQ and original IRX Close/100/252; selector independently matches599 prior-history decisions; ETF wrapper not original futures P&L |
| S6 / causal_cash_v2, original21-session default |Separately demonstrated actual2024–25 TQQQ/TMF60/40 contribution case |
| HFEA weights55UPRO/45TMF |Default21-session actual2024–25 flow comparison reconciles; daily overrides are separately labeled |

The signal arrays are current source vintages used in retrospective research.
Release-time checks are required for a point-in-time certification. Observed
fund costs are already embedded; modeled spread/slippage/commission is external.
Account alternatives do not simulate simultaneous household wash-sale activity.
No advanced derivative or state-tax result is inferred from an ETF unit-price
Monte Carlo convergence pass.

Machine evidence: [cohort gates](evidence/final-v4/final-strategy-cohort.json),
[source catalog](evidence/final-v4/final-source-catalog.json),
[source/provenance manifest](evidence/final-v4/final-manifest.json).


Final21/21 support followed three preserved intermediate cohorts:17 cells before
forward RIC holding proof/signals,18 after Section199A,20 after original VIX/IRX,
then21 after split-adjusted pending wash units. The final same-symbol split
handler preserves sale dollars and original trade shares, proportionally restates
only replacement units, and still rejects unknown different-ticker conversion or
OCC collateral adjustments. Source observation certification remains separate
from execution, household tax completeness and forecast skill.
