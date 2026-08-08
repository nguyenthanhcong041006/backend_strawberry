# Frontend Guide: Strawberry RUL Flutter App

This guide explains how to set up and run the Flutter frontend for the Strawberry RUL Prediction project, especially for Android development with Android Studio.

The frontend is a Flutter app located in:

```text
strawberry_rul_app/
```

It lets the user pick an image from the gallery or camera, sends that image to the FastAPI backend as `multipart/form-data`, and displays the Remaining Useful Life prediction result.

## 1. Frontend Project Location

From the repository root, go to the Flutter app folder:

```powershell
cd your_path...\Strawberry-RUL-prediction\strawberry_rul_app
```

Run all Flutter commands in this folder, not from the repository root.

Important frontend files:

```text
lib/main.dart                                      Flutter app entry point
lib/screens/prediction_screen.dart                Main UI and screen state logic
lib/services/prediction_api_service.dart          Backend API upload service
lib/models/prediction_result.dart                 Backend response parser
pubspec.yaml                                      Flutter dependencies
android/app/src/main/AndroidManifest.xml          Android permissions and app config
android/app/build.gradle.kts                      Android build config
```

## 2. What the App Does

The app has one main screen, `PredictionScreen`, with these states:

```text
initial     No image selected
selected    Image selected from gallery or camera
loading     Waiting for backend prediction
result      Backend returned a valid prediction
invalid     Backend returned an invalid image response
```

The UI is implemented in:

```text
lib/screens/prediction_screen.dart
```

The app uses:

```text
image_picker   Pick image from gallery or camera
http           Send multipart/form-data request to the backend
```

These dependencies are declared in `pubspec.yaml`.

## 3. Install Android Studio

Install Android Studio from the official Android developer website:

```text
https://developer.android.com/studio
```

During setup, install these components:

```text
Android SDK
Android SDK Platform
Android SDK Command-line Tools
Android SDK Build-Tools
Android Emulator
Android Virtual Device
```

After installation, open Android Studio once and finish the setup wizard so that the Android SDK is installed correctly.

## 4. Install Flutter SDK

Install Flutter SDK and add Flutter to your `PATH`.

After installation, check Flutter from PowerShell:

```powershell
flutter --version
```

Then run:

```powershell
flutter doctor
```

For Android development, `flutter doctor` should show Android toolchain support. If it reports missing Android SDK components, open Android Studio and install the missing packages from:

```text
Settings > Languages & Frameworks > Android SDK
```

You may also need to accept Android licenses:

```powershell
flutter doctor --android-licenses
```

## 5. Open the Project in Android Studio

Open Android Studio and choose:

```text
Open
```

Select this folder:

```text
your_path...\Strawberry-RUL-prediction\strawberry_rul_app
```

Do not open only the `android/` folder unless you specifically want to inspect the native Android project. For normal Flutter development, open the `strawberry_rul_app/` folder.

Make sure Android Studio has the Flutter and Dart plugins installed:

```text
Settings > Plugins > Marketplace
```

Install:

```text
Flutter
Dart
```

Android Studio usually installs Dart automatically when Flutter is installed.

## 6. Create or Select an Android Device

You can run the app on either an Android Emulator or a real Android device.

### Android Emulator

In Android Studio, open:

```text
Tools > Device Manager
```

Create a virtual device if none exists. A normal Pixel emulator with a recent Android version is fine.

Start the emulator before running the app.

### Real Android Device

On the phone:

```text
Settings > About phone > Build number
```

Tap `Build number` several times to enable Developer options, then enable:

```text
USB debugging
```

Connect the phone by USB and confirm the debugging prompt on the device.

Check available devices:

```powershell
flutter devices
```

## 7. Restore Flutter Dependencies

From the Flutter app folder:

```powershell
cd your_path...\Strawberry-RUL-prediction\strawberry_rul_app
```

Run:

```powershell
flutter pub get
```

This downloads the packages declared in `pubspec.yaml`, including `image_picker` and `http`.

## 8. Regenerate Flutter Platform Files

If platform files are missing, stale, or were created on another machine, run:

```powershell
flutter create .
```

This regenerates Flutter project scaffolding for the current folder while keeping the existing `lib/` source files.

Use this command from:

```text
strawberry_rul_app/
```

After running `flutter create .`, check that important custom files still contain the expected settings:

```text
android/app/src/main/AndroidManifest.xml
lib/services/prediction_api_service.dart
```

In this project, the Android manifest should include permissions for camera, gallery images, internet access, and cleartext HTTP traffic.

## 9. Backend URL Used by the App

The current frontend backend URL is configured in:

```text
lib/services/prediction_api_service.dart
```

Current value:

```dart
static const String baseUrl = "http://10.0.2.2:8000";
static const String predictEndpoint = "/api/predict";
```

If test on mobile phone:

```text
static const String baseUrl = "http://192.168....your IPV4...:8000";
static const String predictEndpoint = "/api/predict";
```

This means the app sends prediction requests to:

```text
http://10.0.2.2:8000/api/predict
```

If test on mobile phone:

```text
http://192.168....your IPV4...:8000/api/predict
```

Use `10.0.2.2` when the app runs inside an Android Emulator and the backend runs on your computer.

If your backend runs on port `8000`, change the frontend value to:

```dart
static const String baseUrl = "http://10.0.2.2:8000";
```

If you want to keep the frontend as-is, run the backend on port `8001`:

```powershell
cd your_path...\Strawberry-RUL-prediction
py -3.11 -m uvicorn app:app --app-dir src --host 127.0.0.1 --port 8001
```

For a real Android phone, use the LAN IP address of the computer running the backend:

```dart
static const String baseUrl = "http://192.168.1.10:8000";
```

Replace `192.168.1.10` with your real computer IP. The phone and computer must be on the same Wi-Fi network.

For real phone testing, run the backend with:

```powershell
py -3.11 -m uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000
```

## 10. API Contract Expected by the Frontend

The frontend sends the selected image using `multipart/form-data`.

Request field name:

```text
file
```

This matches the FastAPI backend parameter:

```python
file: UploadFile = File(...)
```

Successful backend response expected by `PredictionResult.fromJson()`:

```json
{
  "success": true,
  "remaining_useful_life": 50.0,
  "confidence": 0.8 (model not return conf, you can leave it)
}
```

Invalid image response:

```json
{
  "success": false,
  "message": "Invalid image"
}
```

The frontend parser is implemented in:

```text
lib/models/prediction_result.dart
```

## 11. Run the App

Make sure an emulator or real Android device is available:

```powershell
flutter devices
```

Then run:

```powershell
flutter run
```

If multiple devices are connected, Flutter will ask you to choose one. You can also pass a specific device ID:

```powershell
flutter run -d <device-id>
```

## 12. Recommended First Run Order

Use this order when setting up on a new machine:

```powershell
cd C:\fluttersrc\Strawberry-RUL-prediction\strawberry_rul_app
flutter pub get
flutter create .
flutter run
```

If the app starts but prediction fails, check that the backend is running and that `baseUrl` points to the correct host and port.

## 13. Android Permissions

The Android manifest is:

```text
android/app/src/main/AndroidManifest.xml
```

It currently declares:

```text
android.permission.CAMERA
android.permission.READ_MEDIA_IMAGES
android.permission.READ_EXTERNAL_STORAGE with maxSdkVersion="32"
android.permission.INTERNET
```

The app also sets:

```xml
android:usesCleartextTraffic="true"
```

This is important because the local backend uses plain HTTP, such as:

```text
http://10.0.2.2:8001
```

Without cleartext traffic enabled, Android may block local HTTP API calls.

## 14. Common Problems

### `flutter` command is not recognized

Flutter is not in your `PATH`. Add the Flutter SDK `bin` folder to your environment variables, then reopen PowerShell.

### Android toolchain is missing

Run:

```powershell
flutter doctor
```

Then install the missing Android SDK packages in Android Studio.

### No devices found

Start an Android Emulator from Android Studio Device Manager, or connect a real Android device with USB debugging enabled.

Check again:

```powershell
flutter devices
```

### App opens but prediction fails

Check these points:

```text
Backend is running
Frontend baseUrl uses the correct host and port
Android Emulator uses 10.0.2.2 instead of 127.0.0.1
Real phone uses the computer LAN IP address
Backend endpoint is /api/predict
Multipart field name is file
```

### Image picker does not open camera or gallery

Check Android permissions in `AndroidManifest.xml`, then rebuild the app:

```powershell
flutter clean
flutter pub get
flutter run
```

## 15. Minimal Frontend Checklist

Before testing prediction, confirm:

```text
[ ] Android Studio is installed
[ ] Flutter and Dart plugins are installed in Android Studio
[ ] Flutter SDK is installed and available in PATH
[ ] flutter doctor does not show blocking Android errors
[ ] Android emulator or real Android device is available
[ ] Commands are run inside strawberry_rul_app/
[ ] flutter pub get completed successfully
[ ] flutter create . completed if platform files needed regeneration
[ ] Backend is running on the same host and port configured in baseUrl
[ ] flutter run starts the app successfully
```
