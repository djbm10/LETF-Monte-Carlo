# Support and confidence matrix

As of 2026-11-23. “Available” daily return data is not a validated account backtest. **No full historical after-tax strategy comparison is certified by this audit.** Account APIs run mechanically tested scenarios with explicit inputs; remaining data/tax/execution gates apply.

## Instruments and data

| Instrument/strategy family | Taxable | Roth | Earliest observed input in this audit | Evidence and restriction |
|---|---|---|---|---|
| SPY / regular stock-ETF cash portfolio | Regular federal/NIIT scope tested; state/time gates | Internal trades tax-free; cash-flow history separate | SPY 1993-01-29; other stocks not captured | OBSERVED vendor return data; as-traded fills/distribution payable dates still required |
| TQQQ | Same tax scope; model gate fails | Cash-funded long ETF allowed by generic profile | 2010-02-11 | OBSERVED fund returns; reconstruction fails 2020–25 CAGR gate |
| QLD | Same tax scope | Same | 2006-06-21 | Post-inception proxy passes fixed numerical gate; pre-inception remains hypothetical |
| SSO | Same tax scope | Same | 2006-06-21 | Same; issuer inception differs from first captured trade |
| UPRO | Same tax scope; model gate fails | Same | 2009-06-25 | Reconstruction fails CAGR gate |
| TMF | Same tax scope | Same | 2009-04-16 | Post-inception proxy numerical pass; prehistory uses TLT only from 2002 |
| Diversified/fixed-weight portfolio | Ledger supported | Ledger supported | Latest constituent's genuine start | Membership, covariance and corporate actions must be supplied; no universal historical start |
| Dated qualifying futures | Section 1256 and mechanics tested | Requires dated broker approval/house margins | No independent dated contract tape captured | HYPOTHETICAL settlement/execution scenarios; CME tick math verified only |
| Long options / LEAPS | Integrated tax/exercise gate blocks | Restricted long-quote mechanics with explicit approval | No historical option chains captured | No historical option strategy returns validated |
| Covered calls / CSP / collars / spreads | Complex tax gate; no full execution | Collateral helpers only; assignment/short execution missing | None captured | UNSUPPORTED as a complete strategy |
| KMLM / managed futures | Product tax/data review required | Broker/product review required | No input series captured | Index/constituents investigated; arbitrary drift rejected |
| VBR / small value | Product distributions/data required | Same | No input series captured | Dated benchmark splice investigated, no fabricated early fund history |
| Leveraged gold | Product structure/roll taxation needed | UBTI/product restrictions require review | No dated gold contract tape captured | Spot-gold leverage does not establish futures-product returns |
| Sector rotation | Historical investable universe missing | Same | No constituent universe captured | Generic allocator callback exists; no survivor-free historical comparison established |

## Existing S1–S19 allocation coverage

| IDs | Account-ledger allocation status | Additional gate |
|---|---|---|
| S1 TQQQ, S2 SPY, S3 SSO benchmarks | Implemented conservative cash replay | No claim that original journals establish taxable liquidation returns |
| S4 SMA, S5 SMA band | Implemented prior-close/full-warmup rules; S5 has two-sided hysteresis | Full as-traded prices/distributions and supplied costs |
| S6 TQQQ/TMF portfolio | Fixed-weight replay supported with supplied rebalance frequency | Genuine common data starts 2010; broker/account evidence required |
| S7 volatility target | Implemented cash-constrained target | Target ≤1 of account wealth; not legacy borrowing |
| S9 adaptive volatility | Implemented cash-constrained target | Same; full SMA/vol warmup |
| S10 downside target | Implemented semideviation-based cash variant | This definition/constraint must be distinguished from legacy formula |
| S8 composite; S11–S19 advanced allocators | Explicitly blocked in audited allocation factory | Legacy formulas/journals have not been migrated/reconciled to real fills and cash |

Both account types use the same supported allocation/execution methods. A strategy ID no longer determines whether it is automatically “Roth” or “taxable” inside the audited replay. The caller supplies the account. Unsupported borrowing is rejected instead of charging a negative cash balance the risk-free rate.

## Historical periods and horizons

| Requested period | Evidence available here | Comparison status |
|---|---|---|
| 1970s stagflation | No verified Nasdaq-100/long-bond/derivative inputs for requested funds | UNSUPPORTED as observed fund comparison; use separately sourced benchmark stress models |
| 1987 crash | No captured full constituent benchmarks beginning in 1987 | Same |
| 2000–2012 | SPY/QQQ daily observations; funds observed only after respective starts | Pre-inception rows hypothetical; original future-fit proxies not historical decisions |
| 2008 crisis | QLD/SSO observed; TQQQ/UPRO/TMF not all available | No same-product observed comparison across all strategies |
| 2012–2025 | Observed prices/returns for captured funds | Data supports descriptive fund-return analysis; full after-tax execution still gated |
| 2020 COVID; 2022 rates | Included in frozen reconstruction evaluation | Reported proxy errors, not optimized strategy rankings |
| Mixed/future regimes | Joint paired moving-block paths implemented | HYPOTHETICAL; requires observed inputs, training cutoffs and account-valid events |

The paired generator supports **5, 10, 20, 30 and 40-year** horizons with local seeded RNGs. It does not turn shorter fund histories into observed 40-year returns. Real CAGR requires a supplied aligned CPI series; financing/fund-cost decomposition is `None` when embedded costs cannot be independently measured. Paired confidence intervals apply to independent path outcomes; antithetic pairs must be grouped before resampling. Outcome percentiles are not confidence intervals.

Recurring Roth contributions are tracked as cash flows through `RothHistory`; money-weighted returns are a separate performance API. The current daily strategy replay does not yet automatically schedule all contribution/conversion/withdrawal events. Its lump-sum CAGR must not be used to describe a separate recurring-contribution strategy.
