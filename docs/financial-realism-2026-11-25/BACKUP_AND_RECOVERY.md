# Private backup and recovery procedure

**Status: local bundle and recovery VERIFIED; actual off-host backup NOT VERIFIED.**
The user requires no transfer of licensed/private inputs to unapproved storage.
No approved destination, access/retention policy or transfer mechanism was
provided. No upload was attempted. A destination approval request remains
separate from the completed local engineering work below.

## Prepared artifact and verified scope

Private local archive:
`audit-output/private-recovery/research-v1.tar.gz`

SHA-256:
`e002457010331b9aea760a01286e59dcad97f6aadaac5070fb56a27d6547a707`

Compressed bytes **814,575,466**; uncompressed source bytes **1,344,096,062**;
**551 files**. Sources include the exact macro inputs, original 26 and later 25
forecast registries, two genuinely prospective reference forecasts/commitment,
issuer-tax/index/NAV and original quote sources, and the complete precision
experiment including both large sample binaries. The selection is explicit in
the private `roots.json`; no home directory, cloud configuration, `.aws`, key,
token or credential store is searched. Credential-like filenames, symlinks and
unreviewed file types fail closed. This is a specified research snapshot, not
a whole-machine or uncommitted-user-work backup. Newly generated scenario exports
are independently hashed and reproducible from the preserved sources/code;
they are not asserted to be inside this earlier base archive.

[Creation receipt](evidence/private-bundle-receipt.json),
[restore receipt](evidence/private-restore-receipt.json) and
[semantic recovery](evidence/recovery-semantic-check.json) are public derived
evidence; raw archives/data are private and not embedded in Git. A later local
Git-history bundle/issuer-evidence delta, when recorded in the delivery receipt,
has its own checksum and scope. A Git bundle protects committed refs, not
uncommitted user edits.

The restore drill checks every archived member and all restored bytes. It then
loads **26 + 25 + 2 forecasts**, **53 forecast models**, verifies **52 macro
source files**, and compares every precision sample byte by checksum. Source
catalog hash is exactly
`1d0147b5c242baa5055ba3f9f42e4130ba23dfa2c97287040b2f53c29dcac376`.
Files are created mode 0600, directories 0700. Safe publication uses Linux
`renameat2(RENAME_NOREPLACE)`; existing even-empty targets cannot be replaced.
The supported restore implementation is Linux; other platforms fail explicitly
without that atomic primitive. Temp staging is private and owned by this tool.

Compressed archives are read sequentially. Random extraction in sorted manifest
order can repeatedly decompress gigabytes when physical member order differs.
Streaming validation preserves every hash and catches missing/duplicate members,
traversal, absolute paths, links, sizes and tampering; it does not skip verification.

A separate **11-file issuer/input delta** preserves the newly inspected 2021
PDF, its checked facts/reconciliation and the exact observed TQQQ input bundle.
It is 3,972,231 compressed bytes, SHA-256
`40297127c9f9b1d3edcb0a6975a55b20fadf345d45b96a6f5bbeebcf93b8e1ff`,
at `audit-output/private-recovery/issuer-delta-v1.tar.gz`. Its fresh recovery
reproduces the entire three-case observed-account report exactly.
[Delta receipt](evidence/issuer-delta-recovery.json),
[observed replay after recovery](evidence/restored-observed-reproduction.json).
It is also local only. Preserve both base and delta versions when an approved
off-host destination becomes available.

## Local reproduction and periodic verification

Use the pinned environment from [REPRODUCE.md](REPRODUCE.md). The checksum must
come from the independently retained committed receipt, not solely from an
untrusted file sitting alongside the archive.

```bash
$LETF_PY -m letf.research_backup verify \
  --archive audit-output/private-recovery/research-v1.tar.gz \
  --trusted-sha256 e002457010331b9aea760a01286e59dcad97f6aadaac5070fb56a27d6547a707 \
  --receipt audit-output/new-private-verification.json
$LETF_PY -m letf.research_backup restore \
  --archive audit-output/private-recovery/research-v1.tar.gz \
  --trusted-sha256 e002457010331b9aea760a01286e59dcad97f6aadaac5070fb56a27d6547a707 \
  --target audit-output/new-private-recovery \
  --receipt audit-output/new-private-recovery-receipt.json
```

All target/receipt paths must be new. Restore never uses unrestricted
`extractall`, follows links, replaces existing research or assumes a local
checksum is proof of off-host storage. To create another snapshot, provide a
reviewed JSON object with `roots` (named research directories/files) and
`provenance`, then use `create --roots ... --archive NEW --receipt NEW`.
Any refreshed snapshot has a new creation time/archive identity. Do not overwrite
earlier versions, forecasts or outcomes. Keep provider license/retention notes.

## Required steps before an off-host success claim

1. Approve the specific private destination/account, exact data scope, owner-only
   access, retention/deletion rules, encryption requirements and an existing
   authenticated transfer method. Check source-provider backup/redistribution
   rights. Never provide passwords/private keys in chat or archive them.
2. Retain the committed archive checksum separately. Transfer to a **new versioned
   object**, with private ACLs and appropriate encryption; no public link or
   overwrite. Use only the approved mechanism. The implementation intentionally
   has no auto-selected cloud destination or automatic upload command.
3. Obtain and record a checksum from the actual approved remote object, its
   version/locator and authenticated destination/access evidence. Match the exact
   SHA above. A locally copied file or a user-entered checkbox is insufficient.
4. Read the entire object back through the approved authenticated connection
   into a new private local path. Recompute its checksum and use the restore
   command on that readback. Repeat the semantic source/forecast/sample checks.
5. Record timestamps, destination/object version, access/retention checks,
   remote and readback hashes, recovery results and operator identity in a new
   receipt. Only then may a receipt set `off_host_verified=true`. Immutable/WORM
   storage is a separate property requiring its own evidence.

The current receipts deliberately retain `off_host_verified=false` and no approved
remote destination. Until these steps succeed, losing this host can still lose
the raw research. Git-hosted aggregate forecasts survive local cache loss, but
they are not a substitute for private source/ensemble recovery.
