# Deployment

**Status: the model release is published and the deployment configuration is verified.
The two hosted services are not yet created** — that needs account access.

`v0.2.0` carries the model artifact:
<https://github.com/MRC005/tyre-tread-analysis/releases/tag/v0.2.0>

The build command, the start command, production CORS and model loading were all
dry-run locally against the published release before this was written. The
placeholders marked **[you provide]** are values only the repository owner can supply.

Target topology: **frontend on Vercel, backend on Render.**

## Prerequisites before deploying anything

- [ ] A trained artifact exists in `artifacts/` and `python -m tyretread models` lists it
- [ ] `pytest` passes
- [ ] `python -m tyretread inspect <photo>` gives a sensible verdict locally
- [ ] The frontend works against a locally running backend, on a phone as well as a laptop

Deploying before these hold means debugging the science and the platform at once.

## Backend — Render

A web service built from this repository.

| Setting | Value |
|---|---|
| Environment | Python 3.11 or later |
| Blueprint | `render.yaml` in the repository root — Render can read it directly |
| Build command | see `render.yaml` (installs deps, then fetches the release artifact) |
| Start command | `uvicorn tyretread.api.app:create_app --factory --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |
| Instance type | Free tier is enough to demonstrate; see the cold-start note below |

`--factory` matters: `create_app` is a function, not a module-level `app`, so the
settings are validated at start-up rather than at import.

### Environment variables

| Variable | Value | Why |
|---|---|---|
| `TYRETREAD_ENV` | `production` | Enables the CORS wildcard guard |
| `TYRETREAD_CORS_ORIGINS` | **[you provide]** the Vercel URL | Without it the browser blocks every request |
| `TYRETREAD_MODEL_ID` | the artifact id to serve | Pins the served model |
| `TYRETREAD_LOG_LEVEL` | `INFO` | |
| `TYRETREAD_MAX_UPLOAD_BYTES` | `12582912` | |

No secrets. If that changes, it goes in Render's environment, never in the repository.

### Two things that will bite

**The model artifact has to reach the server.** `artifacts/*.joblib` is gitignored, so
a plain deploy has no model and `/health` reports `degraded`.

**Measured size: 0.72 MB** (750,436 bytes), plus a 14 KB JSON sidecar. That is small
because the estimator is a calibrated ensemble of five RBF-SVMs holding 2,091 support
vectors over 46 float64 features. The size makes a stateless deployment straightforward.

**Recommended: a versioned GitHub release asset, fetched at build time.**

```bash
# Render build command
pip install -r requirements.txt && \
  mkdir -p artifacts && \
  curl -fsSL -o artifacts/$TYRETREAD_MODEL_ID.joblib \
    "https://github.com/MRC005/tyre-tread-analysis/releases/download/$TYRETREAD_MODEL_VERSION/$TYRETREAD_MODEL_ID.joblib" && \
  curl -fsSL -o artifacts/$TYRETREAD_MODEL_ID.json \
    "https://github.com/MRC005/tyre-tread-analysis/releases/download/$TYRETREAD_MODEL_VERSION/$TYRETREAD_MODEL_ID.json"
```

Why this over the alternatives:

| Option | Verdict |
|---|---|
| **Release asset fetched at build** | **Recommended.** Stateless, reproducible — a fresh deploy pins an exact tag, so it obtains exactly the intended model version. No binary in Git history. `curl -f` fails the build loudly rather than deploying a modelless service. |
| Object storage (S3/R2) at build time | Equivalent reproducibility, but adds a credential and a bill for a 0.72 MB file. Worth it only if artifacts become large or private. |
| Render persistent disk | Rejected. Makes the service stateful, ties it to one instance, breaks horizontal scaling, and a disk uploaded by hand is not reproducible — nothing records which version is on it. |
| Commit the `.joblib` to Git | Rejected. Convenient, and 0.72 MB is not fatal on its own, but every retrain adds another permanent copy. Git history is append-only; this compounds. |

The `curl -f` flag matters: without it a 404 writes an HTML error page to
`current.joblib`, and the service starts and reports degraded instead of failing the
build.

Required additional environment variable:

| Variable | Value |
|---|---|
| `TYRETREAD_MODEL_VERSION` | the release tag, e.g. `model-v0.2.0` — **[you provide]** when the release is cut |

**Verify after deploy:** `GET /v1/model` returns the expected `id`, and the metadata
sidecar's `created` timestamp matches the release. A model mismatch is otherwise silent.

**Free-tier instances sleep.** The first request after idling can take 30 seconds or
more, which is fatal in a live demonstration. Either open the app a minute beforehand,
or use a paid instance for the demo.

### Cold-start budget

`opencv-python-headless`, `scikit-image` and `scikit-learn` are a few hundred megabytes
installed. The headless OpenCV build is used specifically to avoid the GUI libraries,
which both bloat the image and fail to install on a slim container. `matplotlib` is in
the `research` extra and is deliberately not a runtime dependency.

## Frontend — Vercel

Not yet built. When it is:

| Setting | Value |
|---|---|
| Framework preset | to be decided with the frontend |
| Environment variable | `VITE_API_BASE_URL` = **[you provide]** the Render URL |

The API base URL must come from the environment, never be hardcoded, so the same build
can point at a local backend or production.

### Camera access requires HTTPS

`navigator.mediaDevices.getUserMedia` is unavailable on plain HTTP except on
`localhost`. Vercel serves HTTPS by default, so production is fine — but testing camera
capture from a phone against a laptop dev server over the LAN will silently fail. Use a
tunnel that terminates TLS, and expect to debug this once.

## After deploying

1. `GET /health` returns `{"status":"ok","model_loaded":true}`
2. `GET /v1/model` reports the expected artifact id
3. An inspection from a laptop browser succeeds
4. **An inspection from a real phone, on mobile data rather than wifi, succeeds** —
   this is the path that matters and the one most likely to break
5. A deliberately bad photograph returns `unable_to_assess` with retake advice
6. A 20 MB upload is rejected with `413`, not a timeout

## What is needed from the repository owner

Nothing yet. When deployment starts:

1. The Render service URL, once created
2. The Vercel deployment URL, once created
3. A GitHub release tag for the model artifact (see the recommendation above), or a
   decision to use a different mechanism
4. Whether a custom domain is wanted

No account will be created, no service configured and no value invented on the owner's
behalf.


---

## Verified before deployment

Each of these was run locally against the published `v0.2.0` release, not assumed:

| Check | Result |
|---|---|
| Release asset downloads anonymously | ✅ 726,706 bytes, SHA-256 matches the local artifact |
| Downloaded artifact loads and predicts | ✅ `svm_rbf_c10`, 46 features |
| Exact Render build command | ✅ fetches both assets into `artifacts/` |
| Exact start command with `TYRETREAD_ENV=production` | ✅ boots, `/health` reports `model_loaded: true` |
| Production CORS with an explicit origin | ✅ `access-control-allow-origin` returned |
| Wildcard CORS in production | ✅ refused at start-up, as designed |

## The served model

| | |
|---|---|
| Artifact | `current` · format v1 · svm_rbf_c10 |
| Release | `v0.2.0` |
| Balanced accuracy | 0.912 ± 0.014 |
| Brier / ECE | 0.069 / 0.066 |
| Abstention rate | 16.6% |
| Decision threshold | 0.51 ±0.2 |

Verify after deploying that `GET /v1/model` reports exactly these numbers. A mismatch
means the build fetched a different release.

## Camera on the deployed origin

`frontend/vercel.json` sets `Permissions-Policy: camera=(self)`. Without it some
browsers block `getUserMedia` on the deployed origin even over HTTPS, which would make
the camera appear broken in production while working locally.
