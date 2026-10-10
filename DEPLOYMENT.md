# Deployment preparation and release gates

**Status: prepared locally; no deployment or repository publication has occurred.**
This application is a research demonstration with three views: building EUI
regression, water-label classification, and Beijing air-pattern clustering.
It is not validated for operational energy, water-safety, or health decisions.

## Release hold: dataset permission

The building and water CSVs are tracked in Git, and the interface can show real
example rows from them. The data owner's permission to redistribute or publicly
display them is unresolved. Keep this gate **on hold**. Do not push this repo to
a public repository, send the complete bundle to a hosting provider, or open a
public app until the owner supplies written permission or an approved replacement
data plan is implemented and checked. An ignore rule does not remove files already
committed to Git history. The Beijing source also needs visible CC BY 4.0
attribution. This document does not assume that permission will be granted.

## Hosting choice

Streamlit Community Cloud is the direct hosting fit for the current Python app.
The app needs a long-lived Python server and WebSocket sessions. A plain Vercel
static or serverless deployment is not a verified fit for this implementation.
Use `app/streamlit_app.py` as the entrypoint, repository-root
`requirements.txt`, and **Python 3.13 selected in Advanced settings**. Keep the
app private initially if the final rights permit that mode, then assess whether
public access is appropriate. No login or role system exists in this source;
any visitor admitted by the host can use all three predictors and Study results.

Do not create a root `uv.lock` from notebook dependencies: Community Cloud may
choose that over `requirements.txt`. The runtime file excludes Jupyter packages.
Streamlit 1.65.0 is proposed because its 2026 releases include newer origin and
input-size hardening. The application has **not** yet been tested with that exact
version. Streamlit 1.57 changed the default server to Starlette/Uvicorn, so the
old direct Tornado pin was removed. This requires a fresh compatibility test.

## Local technical validation before any release

Use a disposable Python 3.13 environment with the **exact** `requirements.txt`
installed, then run from the repository root:

```bash
python -m src.deployment_preflight
python -m unittest discover -s tests -v
python -m src.smoke_check
python -m streamlit run app/streamlit_app.py
```

The preflight is read-only. It checks required files, all 12 station files,
saved outputs in four notebooks, the model/evidence fingerprints, Python 3.13,
and every direct runtime package version. Its `publication_permission` field
will remain `HOLD` even when the technical checks pass. The optional
`--release-gate` exits with an error while that hold remains unresolved. The smoke check loads
trusted model files and makes one source-backed prediction per task; do not run
it on models received from an untrusted party. A model hash beside the model in
the same repository detects mismatches but is not an independent signature.

After those command-line checks, test the **same candidate bundle** in a fresh
browser session: home navigation, all three forms, example-row switches,
blank/invalid/out-of-range refusals, real predictions, charts, Study results,
and JSON downloads. Check desktop and narrow screens. Measure process memory
at startup, after each view, and with a small number of simultaneous sessions;
the local data and model files alone are about 134 MB, which is not a RAM
measurement. Record startup time, peak memory, and any provider limits.

If a model is refitted, rebuild its diagnostics and `results/release_validation.json`
in an isolated candidate copy, rerun the preflight and smoke check, review the
changed metrics and plots, then promote the **entire** reviewed bundle together.
Do not refresh model or evidence files one at a time in a running checkout.
The notebook exports and release-validation script overwrite model/result files
while running, so running them against a live app can temporarily break the
consistency gate. Keep deployed model, evidence, data, and source files read-only
to the app process; only reviewed maintainers should replace a release.

## Host settings to verify on the chosen service

- HTTPS and the effective origin/Host policy, including any reverse-proxy
  forwarded-host behavior. `server.allowedHosts` is available in newer
  Streamlit versions, but its value cannot be set before the final hostname is
  known; verify the host's own routing and set an allowlist where supported.
- Effective Streamlit settings after host overrides: generic browser errors,
  viewer toolbar, CORS/XSRF protection, static serving off, and message limits.
- Filesystem permissions, session/log retention, who can read model/data files,
  and who can change deployments. Avoid logging entered sample values.
- Traffic and concurrency controls, restart behavior, memory limit, and a
  rollback to the previous complete bundle. Model files use `joblib` and must
  come only from the reviewed build.
- Dependency resolution and advisory review for **all resolved packages**,
  not just the direct pins. Retest after any dependency or platform upgrade.

No host configuration can be claimed verified from source alone. The separate
local security review remains a partial source review; it found no demonstrated
browser-reachable exploit and explicitly
left framework routes, exact runtime, and host controls for validation.

## Scientific use limits

The water file's original source, units, and label rules are undocumented, and
its later-year same-file test had lower Poor-class recall than the random
held-out split. The building model's error interval and air cluster stability
also come from the supplied data. New representative data and domain review are
needed before calling this an actual predictor for a real deployment context.

## Sources for hosting behavior

- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit Community Cloud dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Streamlit 2026 release notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2026)
- [Streamlit configuration reference](https://docs.streamlit.io/develop/api-reference/configuration/config.toml)
- [Joblib model-loading warning](https://joblib.readthedocs.io/en/stable/generated/joblib.load.html)
- [Beijing dataset attribution](https://archive.ics.uci.edu/dataset/501/beijing)
