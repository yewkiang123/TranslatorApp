# TextLens (TranslatorApp)

Android app for real-time scene text translation using CameraX, ML Kit OCR/Translate, and OpenCV-based motion gating.

## Features

- Live camera preview with OCR analysis when the frame is stable.
- Source and target language selectors (ML Kit supported language list).
- On-demand translation model download from inside the language dropdown.
- Freeze-frame mode for inspecting a captured frame.
- Save frozen frame to gallery (`Pictures/TextLens` on Android 10+).
- Processing status card (`Ready`, `Detecting Text`, `Translating`, etc.).

## Tech Stack

- Kotlin + Android ViewBinding
- CameraX (`camera-core`, `camera-camera2`, `camera-view`, etc.)
- Google ML Kit
  - Text Recognition (Latin/Chinese/Japanese/Korean)
  - On-device Translate
  - Language Identification
- OpenCV (`org.opencv:opencv:4.9.0`)

## Requirements

- Android Studio (latest stable recommended)
- JDK 11
- Android SDK configured in local environment
- Device/emulator with camera support
- Android `minSdk 24`, `targetSdk 36`, `compileSdk 36`

## Setup

1. Clone or open this folder in Android Studio.
2. Let Gradle sync and download dependencies.
3. Build from Android Studio or CLI:

```bash
./gradlew assembleDebug
```

On Windows PowerShell:

```powershell
.\gradlew.bat assembleDebug
```

## Run

1. Install and launch the app from Android Studio (`app` module) or CLI:

```powershell
.\gradlew.bat installDebug
```

2. Grant permissions when prompted:
- Camera
- Microphone (currently requested by manifest/runtime flow)
- Storage permission on Android 9 and below

## How It Works

- `MainActivity` wires the camera, OCR, translation, tracking, and freeze-frame controllers.
- `OcrService` runs as a CameraX analyzer and only performs OCR when motion is stable.
- `TranslationService` caches translators per source-target pair and downloads language models over Wi-Fi.
- `LanguagePanelController` drives language selection and in-UI model download state.
- `FreezeFrameController` captures current frame, supports pan/zoom, and saves snapshots to gallery.

## Project Structure

```text
app/src/main/java/com/example/translatorapp/
  MainActivity.kt
  camera/CameraController.kt
  ocr/OcrService.kt
  translate/TranslationService.kt
  LanguagePanelController.kt
  FreezeFrameController.kt
```

## Notes

- Translation model downloads require network + Wi-Fi (`DownloadConditions.requireWifi()`).
- If OpenCV fails to initialize, the app shows camera preview but disables OCR.
- Current package/namespace is `com.example.translatorapp`.

## Testing

Run unit tests:

```powershell
.\gradlew.bat test
```

Run instrumentation tests (with connected device/emulator):

```powershell
.\gradlew.bat connectedAndroidTest
```

## Share With A Friend (APK)

Build a shareable debug APK:

```powershell
.\gradlew.bat assembleDebug
```

APK output:

- `app/build/outputs/apk/debug/app-debug.apk`

Optional copy to `dist/`:

```powershell
New-Item -ItemType Directory -Path dist -Force
Copy-Item app\build\outputs\apk\debug\app-debug.apk dist\TextLens-debug.apk -Force
```

Then upload/send the APK file (Google Drive, Dropbox, Telegram, etc.).

Install steps for your friend:

1. Download APK on Android phone.
2. Enable `Install unknown apps` for the app opening the APK (Files/Chrome/etc.).
3. Open APK and tap `Install`.

Notes:

- Debug APK is for direct sharing/testing only.
- For Play Store/public release, use a signed release build (`.aab`/signed `.apk`).
