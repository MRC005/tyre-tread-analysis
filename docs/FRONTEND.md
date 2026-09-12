# Frontend

A mobile-first inspection tool: photograph a tyre, get an assessment, see the reasoning.
React + TypeScript on Vite, no UI framework, no router, no state library.

**55.8 KB of JavaScript gzipped, 4.4 KB of CSS.** Nothing on the critical path is
loaded from a CDN, and there is no webfont request.

---

## Design language

Reference points were professional field-inspection and fleet software, automotive
instrument UI, and document-capture tooling. Nothing was copied — no branding, no
layout, no copy. Three principles were taken from that research and are traceable in
the code:

| Principle from the research | Where it shows up |
|---|---|
| **Never pure white.** `#FFFFFF` is glare on a phone in daylight; a soft neutral ground with white cards reads as paper and separates cards without heavy borders. | `--bg: #f7f8fa`, `--surface: #ffffff` |
| **Status must survive bad conditions.** Inspection tools get used outdoors, at arm's length, sometimes with gloves. | Status colours darkened to clear 4.5:1 on their own tints; 48px targets; glyph + word + colour |
| **Feedback during capture, not after.** Document-capture research is consistent that advice must target the specific defect while the user is still framing. | `lib/liveQuality.ts` → Poor / Fair / Good / Ready with a targeted hint |

### The theme changed from dark to light

The first build was dark. It looked good in a screenshot and was the wrong call for the
product: this is a utility used outdoors, in daylight, next to a car — the condition
where a light interface is legible and a dark one is a mirror. The light theme is now
primary, and the **viewfinder stays dark** because a live camera preview should be
surrounded by neutral dark so the image is judged on its own.

### The accent colour was wrong, and it mattered

An early version used a workshop amber (`#ff8a3d`) for primary actions. It looked right
and was a **functional mistake**: it sat between the caution amber and the alert red, so
the primary button competed with the status colours and diluted them. In a tool whose
entire job is communicating condition, status colour has to be reserved for status.

The accent is a product blue (`#1a63d8`), which cannot be confused with any status.

### Colour contrast

Every foreground/background pair in the palette was measured against WCAG AA. One
failed: muted text at **3.74:1** where 4.5:1 is required, and muted text is used at 12px
for captions, so it is small text by definition. Darkened to `#636e80` — **5.16:1 on
white, 4.85:1 on the page**. All eleven pairs now pass.

### Status is never colour alone

Every status carries **three** signals — colour, a distinct glyph, and a text label —
so it survives colour blindness, greyscale and a bright screen outdoors. Two
components:

- `StatusBadge` — full pill with label, for the result headline.
- `StatusDot` — dot plus a short word, for dense list rows. An early version rendered
  an unlabelled coloured pill in the history list, which read as decoration rather than
  information; a coloured shape with no word is not a status indicator.

### Other decisions worth recording

- **A persistent identity bar.** Without it the home screen read as a marketing page
  rather than the front of a tool.
- **No hand-placed line breaks.** The headline originally used a `<br>` that orphaned a
  word at 375 px and broke differently at every other width. `text-wrap: balance` with a
  `max-width` in `ch` does the job at every size.
- **The idle viewfinder teaches framing.** It was an empty black rectangle, which reads
  as unfinished. It now shows the same framing band the live camera draws, so the user
  learns what is expected *before* the permission prompt.
- **System font stack.** No webfont request on the critical path, and it already looks
  native on every target device.
- **Content is capped, not stretched.** An inspection report is a reading task; a
  1400 px line length is worse, not better.

## Architecture

```
frontend/src/
├── main.tsx            mount
├── App.tsx             the screen state machine, history, health probe
├── api/
│   ├── types.ts        mirrors the backend's Pydantic schemas
│   └── client.ts       fetch wrapper; every failure becomes an actionable sentence
├── lib/
│   ├── camera.ts       getUserMedia, and every way it can fail
│   ├── image.ts        EXIF-correct decode, downscale, re-encode
│   └── history.ts      session-scoped local history
├── components/
│   ├── ui.tsx          Button, Card, StatusBadge, Meter, Disclosure, DataRow
│   └── ui.css
├── screens/            Home · Capture · Preview · Analyzing · Result · Failure · History
└── styles/
    ├── tokens.css      the design system
    └── app.css         screen layout
```

### Why no router

The flow is linear and short, and no step in the middle is meaningfully linkable — a
URL pointing at "analysing", or at a result held only in memory, would be a broken
promise. Screens are an explicit discriminated union in `App.tsx`, which makes every
transition visible in one place and keeps a routing dependency out of the bundle.

### Why no UI framework

The interface is a dozen components. A component library would add more bytes than it
saves and would fight the design language rather than express it.

## The flow

```
Home ──▶ Capture ──▶ Preview ──▶ Analyzing ──▶ Result
          (live        ▲            │            │
         guidance)     └────────────┴── Failure ─┘
```

### The result taxonomy

Four states, each mapping **one-to-one onto a decision the model already validates** —
not an invented four-class model:

| Label | Backend verdict | Meaning |
|---|---|---|
| **Healthy** | `likely_serviceable` | No visible defects found |
| **Attention recommended** | `inconclusive` | Inside the abstention band |
| **Defect detected** | `defect_suspected` | Visible signs of wear or damage |
| **Unable to assess** | `unable_to_assess` | The quality gate refused the photo |

The positive label is deliberately **not "High risk"**. Risk to a driver depends mostly
on remaining tread depth, which this system cannot measure and for which no ground
truth exists in any available dataset. "Defect detected" states what was found; the
recommendation carries the urgency.

### Live capture guidance

`lib/liveQuality.ts` samples the preview at 96×96 every 350 ms and grades it
Poor / Fair / Good / Ready, mirroring the gate's three most common refusal reasons —
brightness, contrast (which separates a dark tyre from an underexposed photo exactly as
the gate does), and sharpness.

It is **a hint, not the gate**. Its thresholds are deliberately *looser* than the
backend's: a live indicator that fires more readily than the real gate would train users
to distrust it, and a preview frame is noisier and lower-resolution than the captured
photograph, so the same numbers would not mean the same thing. The gate remains the
authority.

`Preview` exists so nobody spends mobile bandwidth on a photo they can already see is
wrong. `Failure` is deliberately separate from the "unable to assess" *result*: the
first is the service not working, the second is a valid answer about the image.
Conflating them would tell a user to retake a photograph that was fine.

## Mobile camera

`lib/camera.ts` wraps `getUserMedia` and turns every failure into a title, an
explanation and a route forward.

**The camera is not started on mount.** Requesting permission before the user has asked
for the camera is both worse and riskier — a declined permission is sticky.

**`facingMode: { ideal: "environment" }`**, not `exact`. A laptop has no rear camera,
and an exact constraint would fail outright rather than using the only camera present.

**Resolution is requested deliberately.** The backend refuses an image whose tread band
cannot fill the analysis grid, so the stream asks for 1920×1440; a 640×480 default
would be routinely refused for "resolution too low".

The stream is stopped on unmount and on `visibilitychange`, because a live track keeps
the camera indicator lit and drains battery, and iOS suspends hidden streams in a way
that does not reliably resume.

### Verification status

**Camera capture is implemented but NOT yet verified on physical hardware.** It behaves
correctly in a desktop browser, and every failure path is unit-tested, but neither of
those proves it works on a real phone. See [PHONE_TESTING.md](PHONE_TESTING.md) for the
HTTPS setup and the checklist that would verify it. Until that test passes, this
project does not claim working mobile camera support.

Known browser caveats, none of them yet confirmed against hardware:

| Browser | Caveat |
|---|---|
| iOS Safari | Strictest environment. Requires `playsinline` (set) or video goes fullscreen. Historically lagged on `createImageBitmap`'s `imageOrientation` option — hence the `<img>` fallback. |
| Android Chrome | Generally the smoothest. `facingMode: environment` respected. |
| Firefox mobile | `getUserMedia` supported; rear-camera selection less consistent. |
| In-app browsers (Instagram, etc.) | Often block camera access entirely; the gallery fallback is the route. |

### ⚠️ Camera capture requires HTTPS

`getUserMedia` is unavailable on plain HTTP except on `localhost`. **Opening the dev
server from a phone over a LAN IP will not offer the camera**, no matter how correct
the code is. The app detects this and says so, offering gallery upload instead.

To test camera capture on a real phone, terminate TLS. The project uses a Cloudflare
quick tunnel for this; [PHONE_TESTING.md](PHONE_TESTING.md) has the three commands.

Because the dev server proxies `/api` to the backend, **one tunnel covers both** and
CORS does not apply during the test.

| Failure | What the user sees |
|---|---|
| Not a secure context | "Camera needs a secure connection" + gallery upload |
| No `mediaDevices` | "Camera not available in this browser" + gallery upload |
| `NotAllowedError` | How to re-allow it in site settings + gallery upload |
| `NotFoundError` | "No camera found" + gallery upload |
| `NotReadableError` | "Camera is busy" — another app holds it |

Every branch keeps gallery upload available. Losing camera access must never leave a
user with no way forward.

## Image handling

`lib/image.ts` decodes with `createImageBitmap(..., { imageOrientation: "from-image" })`
so EXIF rotation is applied — phones record orientation in metadata rather than
rotating pixels. Safari has historically lagged on that option, so a plain `<img>`
decode is the fallback.

Two constraints pull against simply compressing hard:

- **Do not over-compress.** The quality gate refuses heavily compressed images, because
  compression destroys the texture the model reads. Re-encoding at quality **0.9** keeps
  measured blockiness near 1.2, against a rejection threshold of 2.2.
- **Do not resize below what the analysis needs.** A **1600 px** long edge matches the
  backend's own reduction and leaves headroom for the oversampling floor. Going smaller
  would manufacture "resolution too low" refusals in the client.

A 12-megapixel phone photo of ~4 MB typically leaves as a few hundred KB.

## API integration

`VITE_API_BASE_URL`, never hardcoded. See `frontend/.env.example`.

| Condition | Code | User sees |
|---|---|---|
| 400 | `undecodable_image` | "That file could not be read as an image." |
| 413 | `file_too_large` | "That photo is too large." |
| 415 | `unsupported_media_type` | "That file type is not supported." |
| 503 | `model_unavailable` | "The analysis service is starting up." *(retryable)* |
| 504 | `inspection_timeout` | "Analysis took too long." *(retryable)* |
| Network / offline | `network` | "Could not reach the analysis service." *(retryable)* |
| Client cancel | `aborted` | nothing — the user chose it |

Screens branch on `code` and never parse a message. A 45-second upload timeout reflects
a large photo on a slow uplink. A raw server message is never shown.

A **health probe on load** lets the home screen warn before the user photographs a tyre
for a service that is down.

## Performance

- Data saver sets `include_evidence=false`, taking the response from ~150 KB to a few
  KB. It **defaults on** when `navigator.connection.saveData` is set or the effective
  type is 2G.
- Evidence panels are `loading="lazy"` inside a collapsed disclosure, so they cost
  nothing until opened.
- Object URLs are revoked on unmount; a long session would otherwise leak megabytes.
- History thumbnails are 96 px, ~2 KB each.

## Accessibility

- Every control is ≥48 px; the focus ring is never removed.
- A skip link precedes the flow.
- Confidence uses `role="meter"` with `aria-valuenow`; evidence bars carry
  screen-reader text since the bar itself is decorative.
- Status is never colour alone — each severity has a distinct glyph and a text label.
- `Disclosure` implements the `aria-expanded` / `aria-controls` pairing.
- Transient status is announced through a polite live region without stealing focus.
- `prefers-reduced-motion` collapses every transition.

## Responsive behaviour

Mobile-first: the base stylesheet is the phone layout, and wider viewports add to it.

| Width | Behaviour |
|---|---|
| < 720 px | Single column, 3:4 viewfinder, full-width buttons, 72 px shutter |
| ≥ 720 px | 4:3 viewfinder, technical panels in two columns, smaller shutter |
| ≥ 1024 px | Content capped at 640 px, capture actions side by side |

Measured at 320, 375, 768 and desktop widths: **no horizontal overflow, no element
wider than the viewport, and no touch target under 44 px** at any of them. At 320 px —
the hardest real case — the primary action is still visible without scrolling.

One defect was found this way and fixed: the app-bar action was 36 px, under the 44 px
floor this project sets for itself.

Content is capped rather than stretched: an inspection report is a reading task, and a
1400 px line length is worse, not better.

## What happens to your photograph

Audited in the source rather than asserted:

| Question | Answer |
|---|---|
| Is the photo sent to the backend? | **Yes** — that is where the analysis runs. |
| Is it written to disk on the server? | **No.** The upload is held in memory, decoded, analysed, and released when the request ends. The only `save()` calls in the package write to an in-memory buffer to encode the evidence panels. |
| Is it stored in a database? | **No database exists.** |
| Is it logged? | **No.** The access log records method, path, status, duration and a request id. No filename, no image data. |
| Does anything persist after the response? | **No.** No background tasks, no temp files, no object storage. |
| What is kept on the phone? | A **96 px thumbnail** (~2 KB) plus the verdict, in `sessionStorage` — cleared when the tab closes. Never the full photograph. |
| Analytics or trackers? | **None.** The page makes no external requests at all: no fonts, no CDN, no analytics. |

The one thing worth naming: in production the web server's access log will contain
client IP addresses, as any web server's does. That is infrastructure logging, not
application behaviour, and no image or inspection content is attached to it.

## Testing

**33 frontend tests** (`npm test`) covering API error mapping, camera error
classification, history and its privacy properties, and the result screen — including
that a refusal does not render as an error, that a sidewall result disclaims tread, and
that no screen claims a tread-depth measurement.

Verified manually against the running backend: upload → analyse → result; blurred and
dark photos → "unable to assess" with specific advice; a sidewall photo → surface
reported with the disclaimer; data saver → no evidence panels; backend stopped →
failure screen; backend restarted → "Try again" recovers without re-selecting the photo.

## Running it

```bash
cp frontend/.env.example frontend/.env.local   # point at your backend
npm install --prefix frontend
npm run dev --prefix frontend
```

With the backend on another port, set `VITE_API_BASE_URL` accordingly:

```bash
python -m uvicorn tyretread.api.app:create_app --factory --port 8010
```
