# Hugging Face Item 1 proxy — rejected for insufficient coverage

## Status

**REJECTED BEFORE RETURN ANALYSIS**

This path used the public `JanosAudran/financial-reports-sec` `large_lite`
Parquet conversion as an independent source of historical 10-K section text.
The experiment was intended to accelerate/replicate the raw-SEC Item 1 network,
not replace its quality requirements.

## Mechanical validation

- all 14 `large_lite` Parquet shards were downloadable;
- total converted Parquet size: 2,028,316,222 bytes;
- schema contained CIK, filing date, document ID, sentence ID, section label,
  and sentence text;
- Item 1 was reconstructed from section class 0;
- sentence ordering was corrected to numeric `sentenceID` order before any
  accepted result;
- the frozen 550-day staleness rule and the same TF-IDF network builder were
  applied.

## Coverage result

The reconstructed corpus yielded only:

- 1,249 valid Item-1 filings;
- 144 unique S&P-related CIKs over 2007-2019;
- quarterly valid Item-1 issuer counts from **61 to 93**;
- median quarterly valid issuer count: **68**.

Frozen network quality gates require:

- minimum quarterly valid Item-1 CIKs >= **200**;
- median quarterly valid Item-1 CIKs >= **350**.

Therefore the HF proxy fails by a wide margin and is **not eligible for strategy
return analysis, model-family advancement, or holdout decisions**.

No Bottleneck return produced from this low-coverage network may be treated as
evidence. The raw SEC EDGAR network build remains authoritative.

## Why this matters

The failure is informative: the public corpus is not a sufficiently complete
historical S&P Item-1 panel for this research design. Relaxing the coverage bar
after seeing the result would create a post-hoc methodology change, so the bar
is retained unchanged.
