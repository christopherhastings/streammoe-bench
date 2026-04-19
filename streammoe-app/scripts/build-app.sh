#!/usr/bin/env bash
#
# Wraps `swift build -c release` output into StreamMoE.app with the
# Info.plist, Resources/Guides, and LSUIElement=1 required for a
# proper menu-bar-only macOS app. Optionally code-signs and notarizes
# when credentials are in the environment.
#
# Usage:
#   scripts/build-app.sh                             # build + unsigned .app
#   CODESIGN_IDENTITY="Developer ID Application: Foo (ABC)" \
#   scripts/build-app.sh                             # build + sign
#   APPLE_ID=me@x APPLE_PASSWORD=app-specific APPLE_TEAM_ID=ABC \
#   NOTARIZE=1 scripts/build-app.sh                  # + notarize + staple
#
# Side-effects: populates build/StreamMoE.app and build/StreamMoE-x.y.zip
# under the repo root. No network access unless NOTARIZE=1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="StreamMoE"
APP_VERSION="${APP_VERSION:-0.1.0}"
BUILD_DIR="$ROOT/build"
APP_DIR="$BUILD_DIR/$APP_NAME.app"
CONTENTS="$APP_DIR/Contents"
MACOS="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"
BUNDLE_ID="com.streammoe.app"

echo "==> clean"
rm -rf "$APP_DIR"
mkdir -p "$MACOS" "$RESOURCES"

echo "==> swift build -c release"
(cd "$ROOT" && swift build -c release --product StreamMoE)
BIN_SRC="$ROOT/.build/release/$APP_NAME"
[[ -x "$BIN_SRC" ]] || { echo "missing release binary: $BIN_SRC"; exit 1; }

echo "==> assemble bundle"
cp "$BIN_SRC" "$MACOS/$APP_NAME"
# Ship the Markdown guides at the expected subdirectory. Bundle.main.url
# in GuideView.swift looks them up as subdirectory:"Guides".
mkdir -p "$RESOURCES/Guides"
cp "$ROOT/Resources/Guides/"*.md "$RESOURCES/Guides/"

cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleExecutable</key><string>$APP_NAME</string>
    <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>$APP_NAME</string>
    <key>CFBundleDisplayName</key><string>$APP_NAME</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$APP_VERSION</string>
    <key>CFBundleVersion</key><string>$APP_VERSION</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <!-- Menu-bar-only: hides the Dock icon + Cmd-Tab entry. -->
    <key>LSUIElement</key><true/>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSHumanReadableCopyright</key><string>Copyright 2026 StreamMoE. All rights reserved.</string>
    <!-- AppDetector shells out to docker/ps. Mark the intent for macOS so the
         user sees sensible consent prompts when a Hardened Runtime sandbox
         is applied at notarization time. -->
    <key>NSAppleEventsUsageDescription</key>
    <string>StreamMoE uses Apple Events to detect whether Open WebUI, Cursor, LM Studio, or Claude Desktop are running so the Connect-an-app pane can show precise copy.</string>
</dict>
</plist>
PLIST

cat > "$CONTENTS/PkgInfo" <<'PKG'
APPL????
PKG

if [[ -n "${CODESIGN_IDENTITY:-}" ]]; then
    echo "==> codesign"
    codesign --force --options runtime --timestamp \
        --sign "$CODESIGN_IDENTITY" "$APP_DIR"
    codesign --verify --deep --strict --verbose=2 "$APP_DIR"
else
    echo "==> skipping codesign (set CODESIGN_IDENTITY to sign)"
fi

ZIP="$BUILD_DIR/$APP_NAME-$APP_VERSION.zip"
rm -f "$ZIP"
(cd "$BUILD_DIR" && /usr/bin/ditto -c -k --sequesterRsrc --keepParent "$APP_NAME.app" "$(basename "$ZIP")")
echo "==> $ZIP"

if [[ "${NOTARIZE:-0}" == "1" ]]; then
    : "${APPLE_ID:?set APPLE_ID}"
    : "${APPLE_PASSWORD:?set APPLE_PASSWORD (app-specific password)}"
    : "${APPLE_TEAM_ID:?set APPLE_TEAM_ID}"
    echo "==> notarize (submit + wait)"
    xcrun notarytool submit "$ZIP" \
        --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" \
        --team-id "$APPLE_TEAM_ID" --wait
    xcrun stapler staple "$APP_DIR"
    # Re-zip after stapling so the distributable carries the ticket.
    rm -f "$ZIP"
    (cd "$BUILD_DIR" && /usr/bin/ditto -c -k --sequesterRsrc --keepParent "$APP_NAME.app" "$(basename "$ZIP")")
    echo "==> notarized + stapled"
fi

echo "==> done: $APP_DIR"
