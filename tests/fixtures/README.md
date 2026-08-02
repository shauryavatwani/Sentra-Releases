# Self-test fixtures

## `selftest_face.jpg`

A **synthetic, AI-generated face** — not a photograph of any real person, and
not connected to anyone enrolled in Sentra. It is never registered, never
logged, and never leaves the self-test.

It is here because `sentra_app.py --selftest` needs to prove that a *built*
Sentra can actually recognise a face, and the only honest way to prove that is
to run a real face through the real pipeline. Every cheaper check was tried
first and every one of them passed on builds that could not recognise anybody:

* checking the model files exist — passed the 1.0.2 Windows build, which died
  inside `model.get()` on missing scipy extensions;
* calling `model.get()` on a blank image — passed the 1.0.3 build on both
  platforms, because `FaceAnalysis.get()` returns early when it finds no face
  and so never reaches the code that was broken.

Running this image through `model.get()` exercises detection, the
`landmark_3d_68` task (which needs `meanshape_68.pkl`), alignment
(skimage → scipy) and the ArcFace recogniser, and asserts a usable 512-d
embedding comes out. A build that cannot recognise anyone cannot pass it.

It is deliberately synthetic because this file ships inside a public installer
and lives in a public repository. A real enrolment photo here would be a
privacy problem no matter how convenient it was.

It is bundled by both `windows/sentra.spec` and `macos/sentra_mac.spec`, which
fail the build if it is missing — the self-test is the gate between a
silently-broken bundle and a customer's machine, so it must not be possible to
weaken it by accident.
