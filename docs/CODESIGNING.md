# StreamMoE — Code Signing & Notarization Requirements

Everything you need in place to ship `StreamMoE.app` outside of
"curl this binary off GitHub" territory. The build script
(`streammoe-app/scripts/build-app.sh`) already has hooks for each step;
this doc is the operator's checklist.

---

## What you need

### 1. Apple Developer account — **paid, $99/year**

- [developer.apple.com/programs](https://developer.apple.com/programs/) — enroll as an individual or organization.
- Individual enrollment takes minutes. Organization enrollment requires a D-U-N-S number and can take days.
- You get: access to certificates, notarization service, App Store Connect.

### 2. Developer ID Application certificate

Used to **sign** the `.app`. This is the cert that ships on end-user
Macs without raising Gatekeeper warnings.

- Issued from Xcode → Settings → Accounts → Manage Certificates → +
  → "Developer ID Application", OR
- Manually from developer.apple.com/account/resources/certificates
- Certificate + private key live in the macOS Keychain on the build machine.
- Check availability: `security find-identity -v -p codesigning` should
  list `Developer ID Application: <Your Name> (TEAMID)`.
- The string before the team ID is what you pass as `CODESIGN_IDENTITY`.

### 3. Notarization credentials (Apple ID + app-specific password)

Used to **notarize** the signed app via Apple's hosted service. Notarization
is a mandatory extra step past signing — without it, Gatekeeper on
Catalina+ will refuse to launch.

- Apple ID with 2FA enabled (same account as the Developer Program).
- App-specific password from [appleid.apple.com](https://appleid.apple.com/) → Sign-In and Security → App-Specific Passwords.
  Create one labeled "StreamMoE notarization". Don't reuse.
- Team ID: find at developer.apple.com/account — 10-character string (e.g. `ABCDE12345`).

These plug into three environment variables:
- `APPLE_ID=you@example.com`
- `APPLE_PASSWORD=xxxx-xxxx-xxxx-xxxx`   (app-specific, NOT your login password)
- `APPLE_TEAM_ID=ABCDE12345`

### 4. `xcrun notarytool` — ships with Xcode Command Line Tools

- `xcode-select --install` if not already installed.
- Sanity-check: `xcrun notarytool --help` should print usage.
- Replaces the deprecated `altool` flow; we target `notarytool` only.

### 5. Hardened Runtime enabled on the build — already done

The `scripts/build-app.sh` passes `--options runtime` to `codesign`,
which opts the binary into Hardened Runtime. Apple's notary rejects
unhardened binaries. Don't remove that flag.

### 6. Entitlements — declarative only if you need them

StreamMoE doesn't currently need custom entitlements (no Mac App Store
sandbox, no custom network permissions, no file-access capabilities).
`Info.plist` already has the two usage-description strings macOS
requires when the bundle exercises:

- `NSAppleEventsUsageDescription` — needed because `AppDetector` shells
  out to `docker ps` and `pgrep`.

If you later add sandboxing, keychain use, camera/microphone, or
network server listening on non-loopback, you'll need a matching
`StreamMoE.entitlements` file and must pass `--entitlements` to
`codesign`.

---

## How the build script uses these

`streammoe-app/scripts/build-app.sh` has three checkpoints:

```sh
# 1. Unsigned build only
scripts/build-app.sh
# output: build/StreamMoE.app + build/StreamMoE-0.1.0.zip (unsigned)

# 2. Signed but not notarized
CODESIGN_IDENTITY="Developer ID Application: Your Name (ABCDE12345)" \
    scripts/build-app.sh
# output: build/StreamMoE.app signed + hardened runtime
# zip is re-created after signing so the distributable carries the signature

# 3. Signed, notarized, stapled
CODESIGN_IDENTITY="Developer ID Application: Your Name (ABCDE12345)" \
APPLE_ID="you@example.com" \
APPLE_PASSWORD="xxxx-xxxx-xxxx-xxxx" \
APPLE_TEAM_ID="ABCDE12345" \
NOTARIZE=1 \
    scripts/build-app.sh
# output: build/StreamMoE.app signed + notarized + stapled ticket
# zip is re-created post-staple so the ticket ships inside the app bundle
```

Notarization takes 2-10 minutes when Apple's service is healthy.
`notarytool submit --wait` blocks until Apple responds.

---

## Distribution options (once notarized)

- **Direct download** — just host the zip on GitHub Releases. Users unzip and drag to Applications. Gatekeeper sees the stapled ticket and launches cleanly. Simplest, no additional infrastructure.
- **DMG with layout** — `create-dmg` or `hdiutil` wraps the notarized `.app` into a disk image with an "drop into Applications" hint. Polish step; not required.
- **Homebrew Cask** — `brew install --cask streammoe`. Requires submitting a cask definition to `Homebrew/homebrew-cask`. Good future move.
- **Mac App Store** — requires a completely different signing identity (`Apple Distribution` cert), sandbox entitlements, review process, and 30 % revenue share. Not the path for a free power-user tool.

---

## Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `errSecInternalComponent` from codesign | cert not in keychain | `security find-identity -v` — if empty, re-download from developer.apple.com |
| "app can't be opened — unidentified developer" on user's Mac | signed but NOT notarized | set `NOTARIZE=1` in the build invocation |
| notarytool returns "Invalid" with logs | Hardened Runtime not applied, or unsigned nested binary | `codesign --verify --deep --strict --verbose=2 StreamMoE.app` to find the offending file |
| "2FA required" during notarize | using Apple ID password instead of app-specific one | generate an app-specific password |
| Stapled ticket doesn't stick | zipped BEFORE stapling | the script already re-zips AFTER stapling; don't manually zip between steps |

---

## Cost & ongoing obligations

- Apple Developer Program: **$99/year**. Lapses auto-invalidate cert.
- Notarization service: **free**, no per-submission cost.
- Typical build+sign+notarize wall time on M4 Max: **~3-6 minutes**.
- When Apple's notary has issues (~1-2 times/year for hours), builds
  are blocked for distribution. Have an unsigned fallback ready for
  internal testers.

---

## Rotation / secrets handling

Do NOT check these into the repo:
- `APPLE_PASSWORD` (app-specific notary password)
- `CODESIGN_IDENTITY` is fine to commit since it's just the cert name
- The private key for the signing cert (never leaves the build machine's keychain)

For CI:
- Store `APPLE_PASSWORD` in the CI secret store (GitHub Actions Secrets, 1Password, etc.)
- Export the signing cert as a `.p12` with a passphrase; store both in secrets; import at job start with `security import`. Delete the keychain entry at job end.
- Keychain unlock with `security unlock-keychain -p "$KC_PW" build.keychain` before signing.

---

## Checklist before first release

- [ ] Apple Developer account enrolled and active
- [ ] Developer ID Application certificate issued and in Keychain
- [ ] `security find-identity -v -p codesigning` shows the cert
- [ ] App-specific password created and stored
- [ ] Team ID captured
- [ ] `xcrun notarytool --help` runs without error
- [ ] Unsigned build succeeds: `scripts/build-app.sh`
- [ ] Signed build succeeds: `CODESIGN_IDENTITY="..." scripts/build-app.sh`
- [ ] `codesign --verify --deep --strict` passes
- [ ] Notarized build succeeds: `NOTARIZE=1 scripts/build-app.sh`
- [ ] `xcrun stapler validate StreamMoE.app` passes
- [ ] Test on a second Mac (different user, Gatekeeper cold) — app launches without warning
- [ ] Upload to GitHub Release and verify download path works end-to-end
