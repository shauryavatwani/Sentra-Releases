# Changelog

Each `## <version>` section below becomes the description of that GitHub
release, and its first paragraph is what Sentra shows in **Settings** before
you install the update. Write it for the person deciding whether to install,
not for whoever wrote the code.

## 1.0.4

Fixes face recognition not working at all — on every installed copy of
Sentra, Windows and macOS both. 1.0.3 only fixed one of two separate causes;
this fixes the one that actually mattered. Also brings one-click updates to
the Mac for the first time.

- **Fixed: nobody was ever recognized on any installed build, on either
  platform.** A reference file InsightFace needs for every single face it
  processes was never included in the app — not since the very first release.
  The live camera feed and the dashboard both looked completely normal, so
  there was nothing to suggest anything was wrong except that nobody was ever
  named. This is different from the Windows-only issue fixed in 1.0.3: that
  one is still fixed and stays fixed, but it was hiding this second, larger
  bug underneath it — on Windows the first bug always failed before recognition
  could reach the second one; on macOS this bug was never actually triggered
  against a live camera before now. The build's own pre-release checks have
  been strengthened to run a face through the full recognition pipeline before
  a build is allowed to ship, so this class of bug cannot reach an install
  again undetected.
- **New: one-click updates now work on Mac.** Previously, clicking "Install
  and restart" on macOS asked for confirmation, then did nothing — the button
  should never have been shown there, since macOS updates could only ever be
  installed by hand. Sentra can now replace itself on macOS the same way it
  already does on Windows: download, verify, install, relaunch, all from the
  Settings page. Your registered faces, detections, visitor records and camera
  settings are untouched either way.
- **Fixed: a failed update attempt showed nothing.** If installing an update
  failed for any reason — an admin declining the Windows permission prompt, a
  permissions problem on Mac, a corrupted download — Sentra showed no error at
  all; the screen just looked like nothing had happened. Failures now show
  exactly what went wrong.

## 1.0.3

Fixes face recognition never working on Windows, even though the camera feed
looked completely normal.

- **Fixed: nobody was ever detected on Windows — no boxes, no "Unknown", just
  the live picture.** The Windows build was missing a runtime component that
  face recognition depends on, so every attempt to identify a face failed
  silently in the background log while the rest of the app looked healthy.
  This never showed up on macOS, which packages that component differently.
- The self-test every build must pass before it can ship now actually runs a
  face through the recognizer, instead of only checking that the model files
  are present. This exact bug could not ship again without that check failing
  first.

## 1.0.2

Fixes the "Install and restart" button, which did nothing.

- **Fixed: installing an update silently failed.** The update downloaded
  correctly, but the installer shut itself down a moment after starting, so
  nothing was ever installed. It was terminating the running copy of Sentra
  *and everything Sentra had started* — which, since Sentra had just started
  the installer, included the installer itself.
- **Fixed: the installer is now asked for administrator permission properly.**
  Windows will show its usual "Do you want to allow this app to make changes?"
  prompt. Choose Yes; declining now gives a clear message instead of failing
  quietly.

If you are stuck on an older version, you can always install by hand from
`C:\ProgramData\Sentra\.updates\` — an installer run from Explorer was
never affected by this.

## 1.0.1

Recognised faces are now picked up without restarting the engine.

- **Fixed: nobody was ever recognised on a new installation.** The list of
  enrolled people was read once when the engine started and never again.
  Because a fresh install ships with nobody registered, that list was always
  empty at startup — so registering someone, or importing a data pack, had no
  effect until the engine was restarted, and nothing said so. The camera
  streamed, the dashboard looked healthy, and faces were simply never named.
  The roster now updates within a few seconds of any change.
- **Fixed: the engine stopped on a brand-new installation** instead of running
  with an empty roster, and pointed at a setup script that does not exist in
  the installed app.
- **Fixed: fight detection was switched off in the shipped build.** Three
  libraries the pose model needs were trimmed from the installer as unused.
  The failure was designed to be non-fatal, so the app ran normally and simply
  never raised an alert.

If you are on 1.0.0 and cannot update yet, **Restart engine** in the top bar
works around the first of these.

## 1.0.0

First packaged release — Windows installer and macOS app.

- One-click install, with everything bundled: no Python, no dependencies.
- Face recognition, multi-camera support, fight detection, and Temporary Pass
  visitor management.
- Built-in updates: Sentra checks on startup and installs from **Settings**,
  keeping your detections, registered faces, visitor records and camera
  settings across upgrades.
- Data packs move a populated system to another machine in one file.
