# SEC bundle preflight request

Purpose: validate that the pinned successful live SEC smoke artifact has at least 10 CIKs with complete, non-degenerate `bottleneck_core` inputs before any QuantConnect cloud compute is launched.

This is an integration/data-coverage check only. It changes no research parameter and reads no return output.

Retry requested after adding the missing workflow-only `requests` dependency. No dataset, signal definition, or research parameter changed.

Credential-presence check requested after the SEC bundle preflight passed. This check does not call QuantConnect and does not print secret values.
