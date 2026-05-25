# Shipping DEM Analyst

This folder contains everything needed to build a portable Windows bundle of
DEM Analyst and ship updates to testers without re-emailing zips every time.

---

## 1. Build a release

```powershell
# from project root (PowerShell)
.\packaging\build.ps1 -Clean        # build only
.\packaging\build.ps1 -Clean -Zip   # also produce dist\DEM-Analyst-<version>-win64.zip
```

What you get:

```
dist\DEM-Analyst\
    DEM-Analyst.exe                <- double-click this
    _internal\                     <- Qt, Python, GDAL, Leaflet, ...
    resources\                     <- map.html, leaflet/, ...
```

The folder is **fully self-contained** — no Python install needed on the
tester's machine. Just copy/unzip it anywhere.

Current size: **~620 MB on disk, ~190 MB zipped**. The bulk is unavoidable:
QtWebEngine alone is ~200 MB (the Chromium engine that hosts Leaflet), and
llvmlite is ~100 MB (Numba's JIT backend used by the hydrology kernels).

If you can drop Numba (the app falls back to pure-Python kernels — slower
but functional), uninstall it from the venv before building and you save
~130 MB.

---

## 2. Why portable .exe (and not an installer)?

**Portable zip** (what the spec produces by default):
- Tester downloads → unzips → runs. No admin rights, no install dialogs.
- Trivial to ship "side by side" versions (`DEM-Analyst-1.2/`, `DEM-Analyst-1.3/`).
- Trivial to clean up — delete the folder.

**Installer** (Inno Setup wrapper around the same bundle):
- Pretty Windows install wizard, Start Menu shortcut, file association
  for `.tif`/`.dem`, Add/Remove Programs entry.
- Mandatory if testers expect a "real" Windows install experience.
- Add later if requested (see §6).

For an internal beta with technical users, portable is simpler. For wider
release, install.

---

## 3. Do you need to keep re-sending the bundle? (No.)

You'll send the **first** copy once. After that, the app updates itself if
you wire in a 50-line "check for updates" routine. The plumbing:

### 3.1 Host the bundle on GitHub Releases

GitHub Releases is a free CDN designed exactly for this. Cut a release:

```powershell
.\packaging\build.ps1 -Clean -Zip
gh release create v1.0.0 dist\DEM-Analyst-1.0.0-win64.zip `
    --title "DEM Analyst 1.0.0" `
    --notes "Initial release"
```

That gives you a stable URL like
`https://github.com/<you>/<repo>/releases/download/v1.0.0/DEM-Analyst-1.0.0-win64.zip`
and a JSON metadata endpoint:
`https://api.github.com/repos/<you>/<repo>/releases/latest`.

Anyone with the link can download — no GitHub account needed for public
repos. For private repos, use a release token or move to a presigned S3 URL.

### 3.2 Add an in-app update check

Drop something like this into `app/core/updater.py` and call it from
`MainWindow.__init__`:

```python
import json, urllib.request, webbrowser
from packaging.version import Version
from app.config import APP_VERSION

LATEST_URL = "https://api.github.com/repos/<you>/<repo>/releases/latest"

def check_for_updates(parent):
    try:
        with urllib.request.urlopen(LATEST_URL, timeout=4) as r:
            data = json.loads(r.read())
        latest = data["tag_name"].lstrip("v")
        if Version(latest) > Version(APP_VERSION):
            from PyQt6.QtWidgets import QMessageBox
            if QMessageBox.question(
                parent,
                "Update available",
                f"DEM Analyst {latest} is available "
                f"(you have {APP_VERSION}). Download now?",
            ) == QMessageBox.StandardButton.Yes:
                webbrowser.open(data["html_url"])
    except Exception:
        pass  # offline / rate-limited — never block startup
```

That's it. Bump `APP_VERSION` in `app/config.py`, build, push a new GitHub
release, and every running tester gets a "new version available" prompt on
next launch.

### 3.3 If you want **silent** auto-updates

If you don't want to make testers re-unzip on every release, look at
[**tufup**](https://github.com/dennisvang/tufup) — it's the modern,
maintained successor to PyUpdater, signs releases with TUF (Update
Framework) so a compromised CDN can't push malicious binaries, and supports
delta patches (download only the changed files, ~5 MB per release instead
of 190 MB).

Worth it once you have more than a handful of testers, or whenever the
"unzip a new folder" friction starts producing support tickets.

---

## 4. The SmartScreen warning

The first time a tester runs an unsigned `.exe` downloaded from the
internet, Windows shows **"Windows protected your PC"** and they have to
click "More info" → "Run anyway". Two ways to defuse this:

1. **Code-signing certificate** (cleanest): buy one from Sectigo / SSL.com
   / DigiCert ($75–$300/year), then sign the bundled exe:
   ```powershell
   signtool sign /a /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 `
       dist\DEM-Analyst\DEM-Analyst.exe
   ```
   SmartScreen still rate-limits new certs ("reputation" needs a few hundred
   downloads to build) but the warning is a single "Run" button instead of
   the scary fullscreen blocker.

2. **Tell testers up front** to expect the warning and how to bypass. Fine
   for a small internal beta.

---

## 5. Release checklist

```
[ ] bump APP_VERSION in app\config.py
[ ] update a CHANGELOG entry
[ ] .\packaging\build.ps1 -Clean -Zip
[ ] smoke-test dist\DEM-Analyst\DEM-Analyst.exe (open a TIF, run hillshade)
[ ] git tag vX.Y.Z && git push --tags
[ ] gh release create vX.Y.Z dist\DEM-Analyst-X.Y.Z-win64.zip
        --title "DEM Analyst X.Y.Z" --notes-file CHANGELOG-X.Y.Z.md
[ ] (optional) sign DEM-Analyst.exe
[ ] post link in tester channel
```

You can collapse the last three lines into a `release.ps1` once the
workflow stabilises. Even better, move the whole thing to GitHub Actions —
push a tag, Actions builds on a clean Windows runner and uploads the
release. Roughly:

```yaml
# .github/workflows/release.yml
on:
  push:
    tags: ['v*']
jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install -r requirements.txt pyinstaller numba
      - run: pyinstaller packaging/DEM-Analyst.spec --noconfirm
      - run: Compress-Archive dist/DEM-Analyst/* "DEM-Analyst-${{github.ref_name}}-win64.zip"
      - uses: softprops/action-gh-release@v2
        with: { files: 'DEM-Analyst-*.zip' }
```

---

## 6. Want a "real" installer instead?

[Inno Setup](https://jrsoftware.org/isinfo.php) wraps the same `dist/`
folder in a single `setup.exe` that creates Start Menu / Desktop shortcuts
and uninstall entries. Roughly 30 lines of `.iss` script. Worth adding the
day a tester says "I can't find it in Start Menu" — not before.

---

## 7. Things to know about the spec

- **UPX is off**. UPX corrupts `QtWebEngineProcess.exe` and several Qt6
  DLLs on Windows — the app crashes on first map render. Don't re-enable.
- **`console=False`**. If you need to debug a crash on a tester's machine,
  flip to `console=True` in `packaging/DEM-Analyst.spec`, rebuild, and ask
  them to run from `cmd` so they can copy the traceback.
- **What we filter post-Analysis**: Chromium DevTools resources (~83 MB),
  all non-English Qt locales (~42 MB), all `.debug.pak`/`.debug.bin`
  variants. If a user reports broken localization, add their locale back to
  `KEEP_LOCALES` in the spec.
- **rasterio `_shim` "ERROR"** during build is harmless — `_shim` was
  removed in rasterio 1.4+ and the hidden import line is just defensive.
- **MSVCR90.dll / tbb12.dll warnings** are also harmless — they're optional
  fallbacks for old freeglut bindings (we use Qt's OpenGL instead) and
  Intel TBB threading (numba falls back to OpenMP/workqueue).
