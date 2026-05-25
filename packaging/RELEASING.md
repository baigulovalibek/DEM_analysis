# Releasing an Update to Users

This is the operational playbook: what to do every time you want to ship a
new version of DEM Analyst to testers.

For background (why portable, why GitHub Releases, what the spec does) see
[`SHIPPING.md`](./SHIPPING.md).

---

## TL;DR — the recurring loop

After the [one-time setup](#one-time-setup) is done, every release looks
like this:

```powershell
# 1. bump the version
#    edit app\config.py: APP_VERSION = "1.2.0"

# 2. build the bundle
.\packaging\build.ps1 -Clean -Zip

# 3. smoke-test
.\dist\DEM-Analyst\DEM-Analyst.exe                  # open it, run hillshade

# 4. commit + tag + push
git add app\config.py
git commit -m "release: 1.2.0"
git tag v1.2.0
git push --follow-tags

# 5. publish the GitHub Release with the zip attached
gh release create v1.2.0 dist\DEM-Analyst-1.2.0-win64.zip `
    --title "DEM Analyst 1.2.0" `
    --notes "What's new in this release..."
```

**That's it.** The next time any tester launches their copy, the in-app
update check sees the new tag and prompts them to download. They never need
you to email them a zip.

If you want it even more hands-off, the [GitHub Actions](#fully-automated-via-github-actions)
variant turns steps 2–5 into "push a tag, walk away".

---

## One-time setup

You only do this once, the first time you release.

### A. Wire the in-app update check into `MainWindow`

Create `app/core/updater.py`:

```python
"""
Lightweight, fail-silent update check.

Compares the local APP_VERSION against the latest GitHub release tag.
If a newer version exists, prompts the user with a download button.
Never blocks startup — any network/parse failure is swallowed.
"""
from __future__ import annotations
import json
import urllib.request
import webbrowser

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QWidget

from app.config import APP_VERSION

# Change this once to point at your repo.
RELEASES_API = "https://api.github.com/repos/<owner>/<repo>/releases/latest"


def _version_tuple(s: str) -> tuple[int, ...]:
    """'1.10.2' -> (1, 10, 2); ignores any leading 'v' or trailing '-beta'."""
    s = s.lstrip("vV").split("-", 1)[0].split("+", 1)[0]
    parts = []
    for p in s.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts) or (0,)


class _CheckThread(QThread):
    found = pyqtSignal(str, str)  # (latest_version, release_html_url)

    def run(self) -> None:
        try:
            req = urllib.request.Request(
                RELEASES_API,
                headers={"Accept": "application/vnd.github+json"},
            )
            with urllib.request.urlopen(req, timeout=4) as r:
                data = json.loads(r.read())
            latest = data.get("tag_name", "")
            if _version_tuple(latest) > _version_tuple(APP_VERSION):
                self.found.emit(latest.lstrip("vV"), data.get("html_url", ""))
        except Exception:
            pass  # offline, rate-limited, repo private — never block startup


def check_for_updates(parent: QWidget) -> None:
    """Fire-and-forget update check. Call once from MainWindow."""
    thread = _CheckThread(parent)

    def _prompt(latest: str, url: str) -> None:
        if QMessageBox.question(
            parent,
            "Update available",
            f"DEM Analyst {latest} is available "
            f"(you have {APP_VERSION}).\n\nDownload the new version?",
        ) == QMessageBox.StandardButton.Yes and url:
            webbrowser.open(url)

    thread.found.connect(_prompt)
    thread.start()
    # Keep a reference so Qt doesn't GC the QThread mid-flight.
    parent._update_thread = thread
```

Then add **one line** to `MainWindow.__init__` (in `app/main_window.py`),
after the UI is set up:

```python
from app.core.updater import check_for_updates
# ... at the end of MainWindow.__init__:
check_for_updates(self)
```

Replace `<owner>/<repo>` in `RELEASES_API` with your actual GitHub path,
e.g. `newprolibek/DEM_analysis`.

### B. Cut the first GitHub Release

```powershell
.\packaging\build.ps1 -Clean -Zip
gh release create v1.0.0 dist\DEM-Analyst-1.0.0-win64.zip `
    --title "DEM Analyst 1.0.0" `
    --notes "Initial release"
```

(You need the [GitHub CLI](https://cli.github.com/) installed: `winget
install GitHub.cli`, then `gh auth login` once.)

### C. Send testers the link **once**

Get the URL from `gh release view v1.0.0 --web` or just copy from the
GitHub UI. Send it. This is the *only* time you'll hand-deliver a link —
from version 1.0.1 onward, the in-app prompt does it for you.

---

## Hotfix flow

Same as the main loop, just bump the patch digit:

```powershell
# app\config.py:  APP_VERSION = "1.2.1"
.\packaging\build.ps1 -Clean -Zip
git commit -am "release: 1.2.1 - fix crash on empty DEM"
git tag v1.2.1 && git push --follow-tags
gh release create v1.2.1 dist\DEM-Analyst-1.2.1-win64.zip `
    --title "DEM Analyst 1.2.1" --notes "Fixes crash on opening DEMs with all-NaN bands."
```

Testers see the prompt on their next launch.

---

## Rollback (something shipped broken)

GitHub Releases are independent of git tags, so you can pull a bad release
without rewriting history:

```powershell
# Hide the release so the in-app check skips it
gh release edit v1.2.1 --prerelease
# or just yank it entirely
gh release delete v1.2.1 --yes
```

The in-app check uses `/releases/latest`, which **excludes prereleases and
drafts**. Marking a release as prerelease is enough to take it out of the
update channel without deleting anything; testers who already pulled it
keep their copy.

To unship completely and roll back to the previous release, also delete
the tag locally and on the remote:

```powershell
git tag -d v1.2.1
git push origin :refs/tags/v1.2.1
```

Then publish a fixed `v1.2.2` (don't reuse `v1.2.1` — it'll confuse caches
and anyone who already downloaded).

---

## Fully automated via GitHub Actions

Once you're tired of running PyInstaller locally, push this to
`.github/workflows/release.yml`:

```yaml
name: release
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
      - run: pyinstaller packaging/DEM-Analyst.spec --noconfirm --clean
      - name: Zip bundle
        run: Compress-Archive dist/DEM-Analyst/* "DEM-Analyst-${{ github.ref_name }}-win64.zip"
      - uses: softprops/action-gh-release@v2
        with:
          files: 'DEM-Analyst-*-win64.zip'
          generate_release_notes: true
```

Then the loop collapses to:

```powershell
# 1. bump version, commit, tag
git commit -am "release: 1.2.0" && git tag v1.2.0 && git push --follow-tags
# 2. ...there is no step 2. Wait ~10 min for the Action.
```

---

## Common gotchas

- **"My update prompt never shows."** Open `app/core/updater.py` and
  confirm `RELEASES_API` points at your real repo. Check the GitHub
  Releases page is public (or testers have a token). Test by temporarily
  setting `APP_VERSION = "0.0.1"` locally and launching.
- **"Tester says SmartScreen blocks the new zip."** Same workaround as
  v1.0.0 — see [`SHIPPING.md`](./SHIPPING.md) §4. Reputation rebuilds
  per-file, so a fresh build *always* triggers the warning on first
  download unless you code-sign.
- **"Zip is too big to attach to a GitHub Release."** GitHub allows 2 GB
  per asset; our bundle is ~190 MB. Should never hit this. If you somehow
  do, host on S3 / Cloudflare R2 and just put the URL in the release
  notes; the in-app prompt only needs `html_url`, which it reads from the
  GitHub API regardless of where the asset lives.
- **"I forgot to bump APP_VERSION."** No-op: testers already on this
  version won't see a prompt. Bump it, retag with the next patch number,
  re-release. Don't reuse a tag.
- **"I want to gate updates to a single tester first."** Mark the release
  as **prerelease** with `gh release edit vX.Y.Z --prerelease` — it stays
  visible on the Releases page but is invisible to `releases/latest`, so
  only people you hand the URL to will see it. Promote with
  `--prerelease=false` once you're happy.
