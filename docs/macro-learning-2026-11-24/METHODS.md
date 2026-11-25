# Model, execution and evidence methods

## Canonical accounting and daily leverage

Observed replay and future exploration both call `run_account_backtest` and
`PortfolioLedger`. Account type is configuration, separate from allocation.
Market generators supply prices/returns, not alternative portfolio balances.
Every daily row reconciles cash, unsettled cash, distribution receivables,
positions and NAV through an independent event replay. Tax-funded sales settle
before payment and generate new current-year realizations. Roth internal sale,
personal withdrawal and contribution are distinct events.

Signals see only history before the execution session. Opens supply nominal
fills with explicit costs; terminal sales use the final close. Splits enter
normalized signal history when known to the replay. Final tax classifications
enter accounting, not trading signals. Assumed daily fills are not NBBO evidence.

For benchmark simple return `r`, leverage `L`, calendar interval `d`, fee `f`,
funding reference `q` and spread `s`, counterfactual daily return is:

```
L * r - f * d / 365 - (L - 1) * (q + s) * d / 360
```

Return factors compound. Path decay is a result, never an extra debit. Fees
and rates accept dated series. A <=−100% return closes the hypothetical fund;
closed units cannot be repurchased. This is a stress convention, not exact issuer
intraday liquidation behavior. Discretionary closure probabilities are unmodeled.
Observed adjusted fund returns instead pass through with leverage one and no
additional internal fee/funding deduction. Taxable fills require nominal prices,
real splits and exact issuer cash, rather than adjusted-close tax lots.

Counterfactual funding annualizes the previous completed month's French RF
return, held until the next month; it is not an actual historical swap invoice.
Monthly data never identify a historical daily path. The 1933 closure uses the
[Fed historical account](https://www.federalreservehistory.org/essays/emergency-banking-act-of-1933)
and [contemporaneous record](https://fraser.stlouisfed.org/files/docs/historical/brookings/17101_1933.pdf).
French observations jump March 3 to March 15. Costs accrue for 12 calendar days;
no missing trading prices are invented. Other extended gaps require evidence.
CAGR includes the first return's previous-close interval. Historical Saturday
observations are retained. CPI purchasing power uses observed monthly endpoints.

## Frozen historical model version

[protocol.json](protocol.json) was committed before evaluation. Source is
`dd460ae`: nine origins, 60 evaluation months, rolling 240-month training,
2,000 paths per origin and seed 20261124 plus declared offsets. Bootstrap needs
36 complete training months; VAR needs 120.

The joint panel contains market log total return, log CPI change, annualized
bill-return short-rate proxy, production log growth and Baa–Aaa yield spread.
Production is not GDP; a par yield is not a bond total return. Confirmed ALFRED
snapshots are preferred. Missing vintages use explicitly revised research data
with a two-month conservative lag and a promotion block. That lag does not
reconstruct revisions. A stochastic bridge spans the last complete information
month to forecast origin without presenting unobserved states as actual data.

| Candidate | Learned information | Fixed assumptions |
|---|---|---|
| Joint moving block | Joint observed rows and within-block dependence | Three-month blocks |
| Regime block | Joint blocks conditional on prior endpoint state | Annualized monthly inflation >4%, negative production growth; joint fallback if matching starts absent |
| Conditional VAR(1)/Student-t | Training-only scaling, ridge coefficients and residual covariance | Ridge 10, df 5, 10% diagonal covariance shrinkage, spectral radius <=0.995 |

No ex-post NBER label enters prediction. Joint historical blocks preserve local
cross-variable dependence; rolling refits allow relationships to change across
origins. VAR covariance is fixed within a fitted version. It is not a complete
dynamic-correlation/causal model, HMM, Bayesian posterior or neural predictor.
Student-t scaling produces the declared covariance. Local RNG sampling does not
force realized paths to match training or evaluation moments. Scenario conditioning
uses the multivariate Student-t conditional mean/scale/df/Mahalanobis adjustment.

All forecasts are content-addressed and stored before outcomes are loaded.
Metadata/array corruption and incomplete artifacts fail verification. This is
local tamper evidence, not externally timestamped write-once storage.

## Proper scores and promotion

CRPS is `mean(abs(X-y)) - 0.5*mean(abs(X-X'))`, normalized by training standard
deviation only. Tests compare the efficient implementation with the independent
pairwise definition. Energy/variogram scores, interval coverage, 5% downside
breaches, drawdown tails, terminal ranks and correlation errors supplement it.
Missing outcome macro values remain missing; joint scores count complete months.
Missing equity intervals cannot be silently dropped. Monthly scoring cannot
establish intramonth tail accuracy. Primary reference:
[Gneiting/Raftery (2007)](https://doi.org/10.1198/016214506000001437).

Complete common eras are selected chronologically without overlap. Six enter
the study: starts 1946, 1970, 1987, 2000, 2008 and 2020. Other declared eras remain
visible. Paired bootstrap confidence intervals resample whole eras and use
97.5% confidence for two candidate comparisons. Six selected nonstationary eras
provide limited inferential power, not a representative random sample of futures.

Promotion requires at least five disjoint eras, registered path counts, >=2%
CRPS improvement, paired mean-loss CI upper <=0, 80% coverage between 65%–95%,
equity 5% breach <=15%, drawdown 95% tail breach in <=30% of eras, verified
information-time sources, unconsumed evidence and baseline/current-incumbent
comparison. Rejections consume selection windows too. Relabeling/revising data
cannot make an inspected period untouched. Historical outcomes are known to
human researchers; these are chronological hindcasts, not real-time old forecasts.

## Later exploratory/prospective version

`adjacent-observation-v2` keeps incomplete months but excludes transitions
involving them; bootstrap blocks must be complete and contiguous. September is
not stitched to December over missing October/November. It does not interpolate
October 2025 CPI or inherit v1's validation. Versions/hashes distinguish artifacts.

Actual prospective issue timestamps are UTC, cannot be backdated and must precede
the first outcome month. Today's archived history can be available today while
remaining invalid for an old origin. Unchanged training/code/protocol fingerprints
produce no retraining. Different candidate cohorts cannot be pooled. Evaluation
accumulates enough complete disjoint eras before selection and requires the
exact current incumbent prospectively frozen at those same issue dates.

The default five-year horizon/five disjoint-era gate needs at least 25 years of
fresh prospective coverage. Monthly issues do not create independent five-year
samples. A shorter protocol needs advance registration. No registry action
changes a trading strategy or deploys a live model.

The optional workflow is off unless repository variables `ENABLE_MACRO_RETRAINING`
and `MACRO_SOURCE_RIGHTS_ACKNOWLEDGED` are both `true`. First initialization is
manual. Later runs require restored history matching the latest successful run;
missing/stale history blocks. Cache/90-day artifacts are not a permanent archive.
Preserve durable copies before activation. This workflow was not enabled here.

## Future scenarios and uncertainty

Monthly equity logs expand into explicitly hypothetical daily Gaussian bridges
conditional on monthly totals, with independently seeded daily variation.
Different paths with the same total produce different leveraged outcomes. An
explicit crash can pin one day's return. This is not an inferred historical path
or calibrated daily-tail model.

Rates, inflation and production inputs condition the joint model. Equity drift
shifts an intercept, not realized performance. `--equity-cagr-constraint 0` fixes
monthly equity total returns at zero. Correlation interventions must preserve a
positive-definite innovation covariance; realized correlation need not equal the
input. The flat-zero-curve bond response remains diagnostic, not a taxable bond
return/ETF sleeve.

Projected XNYS sessions, quarterly ordinary nonqualified distributions, spreads,
slippage, cash sweep `max(reference rate - 0.5%, 0)`, fees and spreads are scenario
assumptions. Distribution yield is annual, paid quarterly. Future tax/contribution
rules freeze an explicitly selected verified 2024/25/26 year, not predicted law.
Owner history is mandatory for flows. No failed account/path is omitted from
comparison percentiles. Internal fee/funding dollar attribution is approximate
opening exposure; modeled NAV deducts these costs only once.

Dated IRR and unitized returns separate cash flows from investment gains.
Ambiguous/undefined IRR remains blocked. Roth terminal internal cash is not an
automatically qualified personal withdrawal. The liquidation probability means
the modeled NAV-floor event, not all real issuer closure risks.

Parameter sensitivity separately resamples joint 12-month training blocks,
excludes join transitions, refits 12 models and simulates 256 paths each. These
unconstrained forecasts begin from the last observed economic state; they are
not the explorer's fixed-6% scenario or bridged 2027 origin. Across-refit median
dispersion is distinct from within-model percentile bands and excludes structural,
source and unseen-regime uncertainty. Probability precision uses
[NIST Wilson intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
Student-t log returns do not have a finite expected arithmetic wealth: quantiles
are reported without an expected-wealth promise.
