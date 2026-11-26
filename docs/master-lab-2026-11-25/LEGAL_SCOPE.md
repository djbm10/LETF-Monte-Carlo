# Legal, account and operational boundaries

This implementation is a bounded research engine, not a complete investor tax
return or broker approval. Official law-source fixtures establish the supported
calculations, not all elections, exceptions, jurisdictions or household facts.
See [historical-law extension](../HISTORICAL_TAX_EXTENSION.md) and the preserved
[tax/data authority catalog](../integration-2026-11-24/TAX_AND_DATA.md).

- ACTUAL_DATED tax uses supported enacted years. MODERN_LAW_COUNTERFACTUAL
  explicitly freezes a stated enacted year against older markets. Future
  simulations freeze a future-law assumption; they do not know future statutes,
  thresholds or investor residency/income.
- Roth internal compliant trading has no routine personal capital-gain tax.
  Cash deposits, eligible contributions, conversions/rollovers and personal
  withdrawals are distinct. Dashboard flows require owner basis/clock/birth,
  MAGI and compensation. Personal conversion/earnings taxes without external
  funding are blocked; no unrestricted margin is silently created.
- Qualified common-share dividends require actual calendar-day holding proof.
  Verified RIC Section199A uses its separate46-day/91-day tracker; preferred/hedged distributions still require separate evidence. Unknown
  eligibility is not automatically reduced-rate income. State fund-distribution
  adjustments need their own source review.
- Alternative taxable/Roth comparisons are not simultaneous household trading.
  External replacement assumptions are explicit. The low-level linked ledger
  models supported cross-account washes, including permanent Roth replacements;
  the portfolio UI cannot infer other owner/spouse activity.
- Zero NAV is not evidence of complete worthlessness under165(g), final
  liquidation, loss character or extinguished recovery rights. The original
  taxable extinction cases stay blocked; no artificial same-day loss or tax
  refund finances another trade. Roth internal closure creates no personal loss
  deduction. See [IRS worthless-security guidance](https://www.irs.gov/faqs/capital-gains-losses-and-sale-of-home/losses-homes-stocks-other-property/losses-homes-stocks-other-property-1).
- Qualifying1256 futures need affirmative contract/tax evidence. Other options,
  straddles, constructive sales, holding suspensions and elections are not
  inferred from a ticker. Actual broker IRA approvals/margin notices, OCC
  deliverables and assignment records remain independent prerequisites.
- A theoretical price, settlement mark, continuous chart, public sample or
  quote checksum cannot certify an executable strategy or its investment returns.

## Private preservation and off-host prerequisites

Current source objects, local archives and restores use private permissions and
trusted SHA-256 verification. Content-addressed extensionless objects are accepted
for backup only when their filename matches the actual bytes. Symlinks,
credential-like names, undeclared members and overwrite targets fail closed.
Git/source preservation is separate from numerical acceptance and investment
prediction. Existing551-file research base/user-patch archives remain untouched;
the new delta and restore receipts are recorded in the final operational manifest.

**No private off-host transfer or off-host restore has occurred.** Local private
permissions are not a claim of cryptographic encryption, WORM storage or a
separately recoverable decryption key. No unapproved cloud account, key or storage
destination was selected. A local restore on the same machine is insufficient.

Before any future off-host action, the user must specify/approve:

1. Exact private provider/account/bucket/path and which licensed/private roots
   may be transferred; provider redistribution/backup entitlements.
2. Encryption/access-control/key-custody rules, with recovery keys separated
   from the archive and credentials handled through an approved secret mechanism.
3. Retention/versioning/immutability and who may read, delete or restore objects.
4. Independent remote checksum/readback and a fresh off-host restore test using
   the trusted manifest, including immutable forecasts and source bytes.

No credential should be pasted into source code, CLI history or public reports.
The optional existing hosted candidate workflow now additionally requires the
approved-private-storage setting. It remains unconfigured here. Ordinary hosted
CI uses authored fixtures and publishes no private raw histories or ensembles.


## Verified local recovery at this delivery

A new private delta preserves1831 files (71,743,728 compressed bytes;
210,395,527 original bytes), including all175 source snapshots,
new immutable candidate/precision artifacts, observed account inputs and a complete
Git branch bundle at `e311bd32bd5db46414670140eaddcd5005bb8b1d`.
Trusted archive SHA-256:`308d50e63b3e9015625a1ae75c9f57ba745f425252c41d33348eacbb5b3a8085`.
A fresh local restore verified every member; code cloned from the restored bundle
and the restored TQQQ2021–25 inputs reproduced taxable216418.1339354625 and
Roth239721.5817711129 **exactly**, with identical pre-tax controls and independent
NAV identities. Source snapshots were not read from their old locations.
[Recovery receipt](evidence/local-recovery.json) records the actual checks.
This supplements the earlier551-file base/user-edit archives. Generated HTML,
old duplicate compressed archives and runtime locks were explicitly inventoried
as exclusions, retained at their original locations, and not treated as inputs.
No encryption or private off-host recovery is claimed.
