# Index Share-Count / Float-Weight Rebalance Pressure — Data Gate

**Disposition: MONITOR — DATA-BLOCKED**

This hypothesis has NOT been return-tested. No holdout has been opened.

## Frozen mechanism
Published index weight change -> passive trackers must alter holdings -> close-auction demand/supply with little fundamental information -> temporary price impact and possible reversal.

Condition X from the mechanism-first backlog:
- official index rebalance changes shares outstanding, float factor/IWF, or constituent weight;
- index membership remains unchanged;
- signed required passive notional is computed from old/new weights using information available before effectiveness.

Primary falsifier:
signed required notional does not predict incremental auction/close pressure relative to unchanged constituents after realistic costs.

## Required PIT input
The test requires, historically and as-known-before-effectiveness:
1. old index shares / IWF / weight;
2. announced new index shares / IWF / pro-forma weight;
3. announcement timestamp and effective date;
4. unchanged membership flag;
5. contemporaneous index-linked AUM estimate or a predeclared scale proxy;
6. ADV / liquidity and prices; closing-auction data is desirable but not strictly required for a daily-close first test.

## Public-source audit

S&P Dow Jones Indices methodology confirms that:
- constituent shares and cap/float factors are updated on scheduled reviews;
- index corporate-event files and pro-forma files communicate upcoming effective index shares/weights in advance;
- quarterly share/IWF changes and accelerated changes have explicit announcement/effective-date rules.

Free public historical S&P datasets located in this pass preserve constituent membership changes, but do not preserve the historical announced pro-forma index-share/IWF fields needed to compute the signed passive notional without reconstruction leakage.

SEC filings can supply issuer shares outstanding and ownership information, but they are not equivalent to the official S&P index shares/IWF actually announced to index clients. Reconstructing the index change from SEC filings would change Condition X and would not be a valid test of this preregistered hypothesis.

## Decision

MONITOR — DATA-BLOCKED.

Do not:
- substitute today's IWF or shares;
- infer old/new index weights from post-effective ETF holdings;
- use current constituent files as historical pro-forma data;
- redefine the hypothesis as generic quarterly share issuance pressure.

The hypothesis can resume only when a PIT archive of S&P (or another specified index provider's) historical pro-forma/corporate-event files is lawfully available. At that point, freeze the sample split and test the original mechanism without changing the signal.

Useful authoritative references:
- S&P DJI methodologies describe quarterly share/IWF updates and advance pro-forma/corporate-event delivery.
- S&P methodology change announcement dated 2018-04-26 documents consolidation of share/IWF implementation timing.
- SEC-filed S&P index methodology descriptions document quarterly/weekly share update rules and pro-forma timing.
