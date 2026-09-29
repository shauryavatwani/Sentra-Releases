# Sentra — CCTV Intelligence

AI CCTV intelligence for institutional security: face recognition, fight
detection and visitor management on your existing cameras.

This repository hosts Sentra's **installers and update feed only**. It
contains no source code. Installed copies of Sentra check this repository's
latest release for updates and install them from inside the app.

## Download

- **Windows (10/11, 64-bit):** [SentraSetup.exe](https://github.com/shauryavatwani/Sentra-Releases/releases/latest/download/SentraSetup.exe)
- **macOS (Apple Silicon):** the `.dmg` in the [latest release](https://github.com/shauryavatwani/Sentra-Releases/releases/latest)

Everything Sentra itself needs is bundled: no Python, nothing to configure.
Neither build is signed with a paid certificate, so the first launch shows a
warning: on Windows choose **More info → Run anyway**; on macOS right-click the
app and choose **Open**.

On first launch Sentra asks to download its face-recognition models (about
290MB, once), showing who made them and the terms they come under; they are not
inside the installer.

## Updates

Installed copies check this repository's latest release for updates and offer
them in **Settings**. An admin clicks Download, then Apply; Sentra verifies the
installer's sha256 digest (published in the release's `version.json`) and
refuses any update that does not match. Releases can require an update
(a minimum supported version) when an older build can no longer be distributed.

Third-party licence notices ship with every build and are shown in
**Settings → About**.

## Copyright

Copyright © 2026 Shaurya Vatwani. All rights reserved. Sentra is proprietary
and licensed for **non-commercial research and evaluation use only**; see
[LICENSE.txt](LICENSE.txt). Operational deployment (for example as a school's
security system) needs written permission and a commercial licence from
InsightFace for its face-recognition models. No copying, redistribution,
reverse engineering, decompiling or disassembling, except where a
third-party component's licence or applicable law permits it.
