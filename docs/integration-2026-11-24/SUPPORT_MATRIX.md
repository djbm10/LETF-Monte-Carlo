# Supported strategy × instrument × account matrix

Status 2026-11-24. “Mechanics supported” is conditional on full source inputs; it is not certification of historic investment returns. Observed inputs and assumed execution have separate labels. No full cross-strategy ranking is authorized.

| Strategy / instrument | Taxable | Roth | Genuine captured start / tested period | Evidence and current gate |
|---|---|---|---|---|
| SMA50 BRK-B | Federal 2024–25 + bounded MA | Generic cash, routine tax zero | Observed 2023-01-03; trading 2024-01-02–2025-12-31 | Real data replay completed; assumed open fills HYPOTHETICAL_STRESS; investable certification BLOCKED |
| Buy/hold TQQQ | Federal-only 2024; full issuer ordinary/QDI payouts | Generic cash | Fund capture 2010-02-11; reviewed account bundle 2024-01-02–12-31 | Real ETF replay completed; QDI investor proof required, state/special components gated |
| Buy/hold QLD | Federal-only 2024 | Generic cash | Fund capture 2006-06-21; reviewed account bundle 2024-01-02–12-31 | Same; no manufactured earlier fund prices |
| SPY, ordinary corporate stocks/ETFs; SMA/bands/fixed weights/vol targets | Shared cash/lot/tax APIs | Same ledger and allocation rule | SPY vendor history 1993-01-29; no complete SPY account bundle certified | As-traded prices, all actions, dated dividends, investor classification and state inputs required |
| SSO | Cash API; actual 2024 tax supplement contains extra components | Cash API; complete reviewed bundle still required | Observed vendor 2006-06-21 | Post-inception numerical proxy PASS; 2024 automatic account builder BLOCKED, including Section 199A components |
| UPRO | Same tax-component gap | Same | Observed vendor 2009-06-25 | Reconstruction CAGR FAIL; account builder BLOCKED on additional distribution tax components |
| TMF | Corporate-fund tax and distribution source required | Cash API | Observed vendor 2009-04-16 | Numerical proxy PASS; no complete account bundle/dated cost schedule certified |
| Fixed diversified portfolio / sector rotation | Cash allocator callback supported | Same | Latest real constituent inception; survivor-free universe not captured | Individual quote availability does not establish historical investable membership or a complete account comparison |
| Frozen 35/0 dated NQ/MNQ futures | Canonical variation/collateral/1256 mechanics | Generic Roth rejects; explicit dated approved profile and house schedule required | Synthetic controlled dated fixture only; no independent CME tape | Research selector reused; hypothetical settlement scenario only; derivative tax funding and dynamic historical margin schedule not complete |
| Long LEAPS/calls/puts | Integrated option tax gate BLOCKED | Approved profile + observed contract quote/depth; restricted long book | Public sample only 2005-09-15, not a price path | Contract parsing/quote/cash mechanics; no certified historical strategy performance |
| Covered calls / CSP / protective puts / collars / spreads | Complex-tax and execution BLOCKED | Collateral helper alone; assignment/exercise combinations BLOCKED | No longitudinal chains captured | Not represented as completed strategies |
| KMLM managed futures / VBR small value / leveraged gold | Source/product-tax gates | Broker/product/UBTI gates | No complete new account input bundle | Dated benchmark constituent/roll vintages required; no arbitrary drift or synthetic prehistory labeled observed |
| Future buy/hold stock bootstrap | Shared daily replay; explicit frozen federal year | Same daily replay | 5/10/20/30/40 × two paths run | HYPOTHETICAL_STRESS; intervals conditional on tiny sample; certification BLOCKED |

The retained legacy strategy definitions are below. The cash-constrained variants in `strategy_allocator` have documented changes (notably downside semideviation and no personal borrowing); they are not silently declared equivalent to every legacy leveraged allocator. No S8/S11–S19 implementation is “brought forward” by renaming it.

| ID / original name | Account allocator | Current blocking requirement |
|---|---|---|
| S1 TQQQ Buy Hold | Implemented | Reviewed 2024 TQQQ federal/Roth example; other dates need full data/taxes |
| S2 SPY Buy Hold | Implemented | Complete distribution/corporate-action input bundle |
| S3 SSO BuyHold (2x) | Implemented | Special distribution components/state treatment |
| S4 200-SMA Simple | Prior-close implemented | Full history and dynamic investor QDI reclassification where needed |
| S5 SMA ±2% Band | Two-sided prior-close hysteresis | Same |
| S6 60/40 TQQQ/TMF | Fixed weights, supplied rebalance schedule | Full common data/distributions from latest inception |
| S7 Vol Targeting 20% | Cash-constrained | No silent personal borrowing; complete inputs |
| S8 Composite Regime | BLOCKED | Individual regime signal and execution/cash reconciliation migration not implemented |
| S9 Adaptive Vol Target | Cash-constrained | Distinguish from legacy borrowing; full SMA/vol warmup |
| S10 Sortino Optimize | Cash-constrained semideviation | Definition differs from legacy truncated-sample statistic; no equivalence claim |
| S11 Hyper-Convex | BLOCKED | Nonlinear leverage sizing/financing/account reconciliation |
| S12 Vol-Velocity | BLOCKED | Velocity signal causality, permitted exposure and trade reconciliation |
| S13 VoV Momentum | BLOCKED | VoV estimator/warmup/causal weighting and fills |
| S14 Skewness-Adjusted | BLOCKED | Skew estimator robustness and account leverage/financing |
| S15 Meta-Ensemble | BLOCKED | Causal component state, aggregation and canonical order mapping |
| S16 Crisis Alpha | BLOCKED | Crisis-state observability, instrument availability and ledger migration |
| S17 Tail Risk Optimizer | BLOCKED | Tail estimates, sizing, crash/liquidation mechanics and account migration |
| S18 Mom. Vol Conv. | BLOCKED | Momentum/volatility rule and turnover reconciliation |
| S19 Conviction Compounder | BLOCKED | Conviction state, permitted position sizing and full execution migration |

| Historical environment | What exists | Released comparison status |
|---|---|---|
| 1970s stagflation | Research factor inputs; no genuine requested fund history or dated personal tax support | BLOCKED as observed/account comparison |
| 1987 | No complete captured requested product/derivative histories | BLOCKED; daily benchmark-based crash scenarios must be labeled hypothetical |
| 2000–2012 / 2008 | SPY/QQQ inputs; some funds after genuine inception | No all-product observed ranking; pre-inception daily proxies hypothetical |
| 2012–2025 | All five captured funds have vendor observations | Descriptive/reconstruction errors available; account bundles only stated subperiods |
| 2020 and 2022 | Included in unchanged reconstruction evaluation | Error reports published; failed gates retained |
| Mixed future regimes / closures / financing/correlation shocks | Generic paired generation and basic gaps; not a fully implemented joint stress framework | BLOCKED as comprehensive comparisons |

Current tax parameters do not support historical years before 2024. Genuine data start dates cannot override that legal-model gate. Financing/fund fees embedded in observed returns are not itemized as independently measured dollars; such decompositions remain unavailable rather than fabricated zeros. Source/model uncertainty is disclosed; the short demo and two-path bootstrap do not quantify it fully.
