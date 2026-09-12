# Testing on a real phone

Camera capture needs HTTPS. `getUserMedia` is unavailable over plain HTTP except on
`localhost`, so opening the dev server from a phone on a LAN IP will **never** offer the
camera — no code change fixes that. This sets up a temporary HTTPS address instead.

**One tunnel is enough.** The Vite dev server proxies `/api` to the backend, so the
phone sees a single origin and CORS does not apply. There is nothing to configure.

---

## Three commands

Each runs in its own terminal tab, from the project root. Leave all three running.

**1 — backend**

```bash
python -m uvicorn tyretread.api.app:create_app --factory --host 127.0.0.1 --port 8010
```

**2 — frontend**

```bash
npm run dev --prefix frontend
```

**3 — HTTPS tunnel**

```bash
cloudflared tunnel --url http://localhost:5173
```

If `cloudflared` is not installed:

```bash
brew install cloudflared
```

No account, no signup, no configuration.

### The URL to open on your phone

The third command prints a box a few seconds after starting:

```
+--------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at ...                |
|  https://something-random-words.trycloudflare.com                   |
+--------------------------------------------------------------------+
```

**Open that `https://…trycloudflare.com` address on your phone.** That is the whole
setup. The address changes every time you restart the tunnel.

### What about CORS?

**Nothing to set.** The frontend calls `/api/...` on its own origin and Vite forwards it
to the backend, so the browser never makes a cross-origin request. `TYRETREAD_CORS_ORIGINS`
is irrelevant for this test — it matters only for the eventual Vercel/Render split, where
the two really are on different hosts.

---

## Test checklist

Roughly five minutes. What to look for is as important as what to do.

### A · Open the URL
Page loads over HTTPS, padlock shown. The header reads **TreadCheck**.
⚠️ If it says *"Camera needs HTTPS"* you have opened the LAN address, not the tunnel.

### B · Start inspection
Tap **Start inspection**. You should land on *Take a photo* showing a **framing diagram**
— dashed band, groove slots, blue corner brackets. That is the empty state, not a bug.

### C · Allow camera
Tap **Open camera** → the browser asks for permission → **Allow**.

- ✅ The **rear** camera should open, not the selfie camera.
- ✅ A dashed band with *"Fill this band with the tread"* appears over the live view.
- 🔍 Check the image is not stretched or squashed.

**Also test declining:** reload, tap **Open camera**, tap **Block**. You should get
*"Camera permission was declined"* with instructions and **Choose a photo** still working
— never a dead end.

### D · Capture a tyre
Fill the dashed band with the tread. Tap the white shutter button.

- 🔍 The shutter should be reachable with your thumb one-handed.
- 🔍 The camera indicator should switch off immediately after capture.

### E · Preview
- ✅ The photo appears **the right way up** — not rotated or mirrored. This is the EXIF
  test and the one most likely to fail on a specific phone.
- ✅ Below it: dimensions and something like *"420 KB (resized from 3.8 MB)"*.

### F · Analyse
Tap **Analyse this photo**.

- ✅ Five progress steps appear.
- ✅ On mobile data it should finish in a few seconds. Under ~2 s on wifi.

### G · Result
- ✅ A verdict, a confidence bar, **Surface assessed**, **Image quality**, **What to do**.
- ✅ Readable at arm's length without zooming.
- 🔍 Scroll the whole way: no horizontal scrolling, nothing cut off by the notch or the
  home indicator, nothing hidden behind the address bar as it collapses.

### H · Technical analysis
Tap **Technical analysis ▾**.

- ✅ Five image panels: original with a detected-area box, contrast-equalised, analysed
  region, edge map, frequency spectrum.
- ✅ Quality checks with ✓/✕, feature measurements, a **Diagnostics** block noting
  `legacy_tsci` is *not used by the model*, and model details.
- 🔍 Panels should load without a long blank gap.

### I · Retake
Tap **Retake this photo** → back to capture. Camera should reopen cleanly.

### J · A deliberately bad photo
Photograph something very dark, or cover the lens most of the way.

- ✅ Expect **"Unable to assess"** with a specific reason (*"Photo is too dark"*) and a
  fix (*"Move somewhere brighter…"*).
- ✅ It must **not** look like an error — no red alarm styling, no stack trace, and no
  confidence bar. One clear **Retake photo** button.

### K · A sidewall photo
Photograph the **side** of the tyre — the lettered wall, not the tread.

- ✅ A verdict still appears, but **Surface assessed: Sidewall / shoulder**.
- ✅ An amber note: *"…it says nothing about how much tread is left."*

This is the honest-scope behaviour: the system will assess the rubber it can see, and
tells you it is not a tread assessment.

### L · Another tyre
Tap **Inspect another tyre**, repeat. Go **Home** — the previous inspections appear under
**This session** with thumbnails and status dots.

---

## Expected results at a glance

| Test | Expected |
|---|---|
| A | Loads over HTTPS, no camera warning |
| B | Framing diagram visible |
| C | **Rear** camera, framing band; declining gives advice + upload fallback |
| D | Shutter thumb-reachable; camera light off after capture |
| E | **Photo upright**; resized-size line shown |
| F | Progress steps; a few seconds |
| G | Verdict + confidence + surface + quality + action; no horizontal scroll |
| H | 5 panels, quality checks, diagnostics marked unused, model details |
| I | Returns to capture cleanly |
| J | **Unable to assess** + reason + fix; not styled as an error |
| K | Verdict + **Sidewall / shoulder** + tread disclaimer |
| L | History shows previous inspections |

## What to send back

For anything that looks wrong: **which step**, **what you saw**, and **your phone and
browser** (e.g. "iPhone 14, Safari"). A screenshot helps most for layout problems.

The likeliest genuine failures are **E** (EXIF orientation differs by phone) and layout
issues on a notched device — both are device-specific and cannot be reproduced here.

## Known limitations before you start

- **Camera capture is unverified on physical hardware.** It is implemented and behaves
  correctly in a desktop browser; this test is what would verify it.
- The free tunnel is slower than production will be — a couple of seconds of latency is
  the tunnel, not the analysis (the backend takes ~80–200 ms).
- The tunnel URL dies when you stop `cloudflared`, and the next one is different.
- iOS Safari is the strictest environment for camera capture; if anything fails, it is
  the most likely place.
