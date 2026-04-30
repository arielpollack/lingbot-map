# lingbot_capture

Slim Flutter POC: record a video on the phone, upload it to the deployed
LingBot-Map backend, and view the resulting 3D model.

The backend lives on Modal at:

    https://arielpollack--lingbot-map-web-fastapi-app.modal.run

(Hard-coded as `kBackendBase` in [lib/main.dart](lib/main.dart). Change there
if you redeploy under a different workspace/app name.)

## Flow

1. **REC / STOP** — captures video with the rear camera (no audio).
2. **UPLOAD** — `POST /api/runs` multipart with `video=...&fps=10&mode=streaming`.
3. App polls `GET /api/runs/{id}` every 5 s.
4. On `completed`:
   - **VIEW IN APP** — opens the textured-mesh viewer in an in-app WebView.
   - **OPEN IN BROWSER** — same URL via the system browser.
   - **COPY URL** — copy to clipboard (paste into a desktop browser for the
     full Three.js / mkkellogg viewer experience).

`/mesh?run=...` is preferred when a textured mesh exists; otherwise the app
falls back to `/viewer?run=...` (point cloud).

## Run on a real iPhone (the simulator has no camera)

### Option A — Xcode (one-time signing setup)

```sh
SSL_CERT_FILE=/Users/ariel/develop/lingbot-map/poc/.data/ca-bundle.pem \
  flutter pub get
open ios/Runner.xcworkspace
```

In Xcode:

1. Select the **Runner** project, then the **Runner** target → **Signing & Capabilities**.
2. Set **Team** to your personal Apple-ID team.
3. Change **Bundle Identifier** to something globally unique
   (`com.lingbot.lingbotCapture.<yourname>` works fine).
4. Plug in the iPhone, select it as the run destination, ▶︎ **Run**.

The first launch will fail with "untrusted developer". On the phone:
**Settings → General → VPN & Device Management → trust your dev cert.**

### Option B — `flutter run`

```sh
flutter devices                       # find the iPhone's id
flutter run --release -d <iphone-id>
```

Requires the same one-time Xcode signing config (Apple won't let `flutter run`
install on a physical device without a configured signing team).

## Permissions

Set in [ios/Runner/Info.plist](ios/Runner/Info.plist):

- `NSCameraUsageDescription`
- `NSMicrophoneUsageDescription` (the camera framework asks for it even though
  audio is disabled)
- `NSPhotoLibraryUsageDescription`

## Network through the Monday corporate proxy

`pub get` and image fetches go through the proxy and need the bundled CA. The
backend itself is `*.modal.run` — that is on the proxy allow-list and works
from the phone over LTE/Wi-Fi without any extra config.

If `flutter pub get` ever hangs:

```sh
SSL_CERT_FILE=/Users/ariel/develop/lingbot-map/poc/.data/ca-bundle.pem \
  flutter pub get
```

## Redeploying the backend

The backend image, secrets, and Dict store are managed in
[`poc/app/modal_web.py`](../poc/app/modal_web.py).

```sh
modal deploy poc/app/modal_web.py
```

The URL is stable across deploys (workspace + app name determine it).
