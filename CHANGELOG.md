# Changelog

Each `## <version>` section below becomes the description of that GitHub
release, and its first paragraph is what Sentra shows in **Settings** before
you install the update. Write it for the person deciding whether to install,
not for whoever wrote the code.

## 1.0.7

Stops fight detection raising false alarms about people who are simply standing
close together. A hug, a hand on someone's shoulder, or two people talking at
arm's length could be reported as a fight; they no longer are.

- **Fixed: hugs and other close contact reported as fights.** Sentra scored a
  pair of people on three things — how close together they are, how fast their
  arms are moving, and whether a hand is reaching into the other person's space
  — and it was possible for the first and third to add up to an alert on their
  own, with the arm-movement score contributing nothing at all. Those two are
  really the same observation twice (people standing close together also have
  their hands near each other), so anyone embracing, being helped up, or having
  their shoulder squeezed scored the same as someone throwing a punch. Fight
  alerts now require actual movement: closeness alone can no longer raise one,
  no matter how close.
- **Fixed: a still camera image slowly counting as movement.** The arm-movement
  score adds up every bit of motion it sees between frames, and the underlying
  pose estimate wobbles by a pixel or two even when a person is standing
  perfectly still. That wobble accumulated, so a motionless person gradually
  read as gently moving. It is now ignored, which means the movement score
  reflects real movement.
- **Real fights are detected exactly as before.** This release only removes
  false alarms — every fight the previous version would have alerted on still
  alerts, verified side by side against the old code.

## 1.0.6

Adds accounts for the rest of the team, lets anyone change their own password,
and fixes a camera that could stay stuck on "Offline" after the network came
back.

- **New: change your own password.** Settings now has a *Change your password*
  panel. It asks for your current password first, and the change is saved
  properly — it survives restarting Sentra and installing an update.
- **New team logins.** Three more management accounts have been added.
- **Fixed: a camera could stay stuck on "Offline" after coming back.** If the
  network dropped, the camera rebooted, or Wi-Fi blipped, Sentra kept trying to
  read from a connection that was already dead and never re-established it — so
  a camera that was genuinely back online still showed as Offline until Sentra
  was restarted by hand. It now reconnects on its own.
- **Fixed: re-issuing a pass for a visitor whose photo had been deleted.**
  Visitor photos are deleted automatically a while after a pass ends. Re-issuing
  a pass after that point would have re-admitted someone with no photo left to
  check them against, so it is now refused with an explanation, and the button
  is replaced by *Photo expired*. Re-issuing still works normally while the
  photo is there.
- **Retiring old versions.** Sentra can now be told that a version is no longer
  supported. On a version that has been retired, the dashboard is replaced by a
  page explaining that an update is required — cameras and face recognition keep
  running in the background, and no data is touched. This takes effect for
  versions from 1.0.6 onward; it cannot be applied to versions already
  installed before this release.

## 1.0.5

Makes updating on the Mac actually work end to end, and cuts about 180 MB off
the download on both platforms.

- **Fixed: updating on a Mac left Sentra not running, and macOS then said
  "Sentra is not open anymore".** The new copy was being launched a moment
  before the old one had finished shutting down, so it found the old one still
  holding the dashboard, assumed Sentra was already running, and quit straight
  away — leaving nothing running at all. The relaunch now waits for the old
  copy to fully release the dashboard first.
- **Fixed: updating on a Mac left the old camera engine running.** It kept hold
  of the camera after the update, so the new copy could not connect to it. The
  engine is now stopped properly as part of installing.
- **Fixed: Sentra failing to start on a Mac showed nothing at all.** Because it
  runs in the background with no window there, a startup problem — most often
  another program already using port 8000 — produced no message anywhere.
  Sentra now shows a real dialog explaining what happened, as it already did
  on Windows.
- **Smaller download.** Removed a large data library (~180 MB) that came in
  with the fight-detection package but is only ever used for training models,
  which Sentra never does. Verified by running the full detection pipeline
  without it.
- Every build must now recognise an actual face before it is allowed to ship.
  The check runs a real face through the complete recognition pipeline inside
  the finished app; previous versions only checked that the model files were
  present, which is how a build that could not recognise anyone shipped twice.

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
