# Changelog

Each `## <version>` section below becomes the description of that GitHub
release, and its first paragraph is what Sentra shows in **Settings** before
you install the update. Write it for the person deciding whether to install,
not for whoever wrote the code.

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
