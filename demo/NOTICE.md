# Notices

The viewer in this directory is a local copy of the prebuilt demo from
[adamraudonis/splats4D](https://github.com/adamraudonis/splats4D), which is
released under the MIT License. The license text is in `LICENSE`.

The renderer is a line-by-line port of the
[antimatter15/splat](https://github.com/antimatter15/splat) WebGL viewer.
The `flame.splat4d` and `hero.splat4d` scenes are the files published in that
repository's `docs/demo/` directory.

This copy hides the encode panel and the compare control. Encoding requires
the upstream Rust CLI and the original frame directory, which are not part of
this talk demo. Compare mode requires the raw `.splat` frames.
