# Deployment steps

Two services. Render first, because the frontend needs the backend's URL.

---

## Step 1 · Render (backend)

1. Go to **[render.com](https://render.com)** → sign in with GitHub.
2. **New** → **Blueprint**.
3. Connect the repository **`MRC005/tyre-tread-analysis`**.
   Render finds `render.yaml` automatically and pre-fills everything.
4. It will prompt for one value it cannot guess:

   | Variable | Enter for now |
   |---|---|
   | `TYRETREAD_CORS_ORIGINS` | `https://placeholder.vercel.app` |

   The real Vercel URL does not exist yet — we correct this in step 3. The service
   refuses to start with `*` in production, so a placeholder is needed rather than a
   wildcard.
5. **Apply** / **Create**. First build takes roughly 3–6 minutes (installing OpenCV and
   scikit-learn dominates).

**Send me the service URL** — it looks like `https://treadcheck-api.onrender.com`.

I will then verify `/health`, `/v1/model`, and run a real inspection against it.

> The blueprint sets region `singapore`, the closest Render region to India. Change it
> in `render.yaml` if you'd prefer elsewhere.

---

## Step 2 · Vercel (frontend)

1. Go to **[vercel.com](https://vercel.com)** → sign in with GitHub.
2. **Add New** → **Project** → import **`MRC005/tyre-tread-analysis`**.
3. Set these, then deploy:

   | Setting | Value |
   |---|---|
   | **Root Directory** | `frontend` ← *easy to miss, and it fails without it* |
   | Framework Preset | Vite (detected automatically) |
   | Environment Variable | `VITE_API_BASE_URL` = the Render URL from step 1 |

   Enter the Render URL with **no trailing slash**.

**Send me the deployment URL** — it looks like `https://tyre-tread-analysis.vercel.app`.

---

## Step 3 · Close the loop

Back in Render → your service → **Environment**, replace the placeholder:

| Variable | Value |
|---|---|
| `TYRETREAD_CORS_ORIGINS` | the Vercel URL from step 2 |

Save. Render redeploys automatically (about a minute).

Without this the browser blocks every request from the real frontend.

---

## Then I run the production smoke tests

Desktop upload, a real inspection, unable-to-assess, sidewall, oversized file, invalid
file, backend-unavailable behaviour, and the served model version — against the live
URLs, not locally.

You run the one test I cannot: **the camera on your phone**, over the Vercel HTTPS URL.

---

## Free-tier cold starts — worth knowing before a demo

Render's free tier **sleeps after 15 minutes idle**. The first request then takes
roughly 30–60 seconds while the container wakes.

The app handles this correctly — the health probe marks the service unreachable and the
error screen offers a retry rather than hanging — but a 45-second wait in front of an
interviewer is bad.

Pick whichever fits:

| Option | Cost | Notes |
|---|---|---|
| **Open the app a minute before demoing** | free | Reliable if you remember. What I'd do. |
| An uptime pinger hitting `/health` every 10 min | free | e.g. UptimeRobot. Keeps it warm; mild misuse of a free tier. |
| Render Starter instance | ~$7/month | No sleeping. Worth it around interview season. |
