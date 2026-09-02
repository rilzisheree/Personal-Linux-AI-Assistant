---
name: Current currency endpoint
description: Live no-key exchange-rate provider behavior for Lura information tools.
---

Use ExchangeRate-API's public v6 latest endpoint with the base currency in the
path. It supports SAR and returns a provider update timestamp alongside rates.

**Why:** The older Frankfurter endpoint redirected to a v1 URL and did not
support the requested SAR conversion path, causing a live 404 despite mocked
tests passing.

**How to apply:** Treat exchange rates as live data, validate the provider
result and requested currency, and return the provider's update timestamp with
the converted amount. Re-check the endpoint contract if the provider changes.