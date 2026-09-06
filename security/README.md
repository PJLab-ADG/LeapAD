# Security notes (LeapAD)

## Semgrep finding on `tools/fast_api_vlm.py` (path traversal)

**False positive.** Line ~67 uses `Image.open(io.BytesIO(...))` on base64 image
bytes from JSON. No filesystem path is taken from the HTTP client. Temporary
filenames are built from `args.tmp_dir` (CLI) + `time.time()`.

## Real issue: OpenSCENARIO command injection (Critical)

`openscenario_parser.py` maps `CustomCommandAction/@type` into `RunScript`,
which previously executed the string with `subprocess.Popen(..., shell=True)`.

A malicious `.xosc` can therefore run arbitrary shell commands when a scenario
is loaded (e.g. `python /path/to/noop.py ; touch /tmp/pwned`).

### Reproduce

```bash
python security/poc_runscript_shell_injection.py
# Expect: VULNERABLE on unpatched tree
```

### Fix

`RunScript` now uses `shlex.split`, `shell=False`, and rejects script paths
outside `base_path`.

## Real issue: XXE via `xml.etree.ElementTree.parse`

Route / OpenSCENARIO / scenario XML was parsed with the stdlib ElementTree API,
which can resolve external entities depending on the Python build.

### Fix

Introduced `srunner/tools/safe_xml.py` (prefers `defusedxml`, otherwise rejects
`DOCTYPE`/`ENTITY`) and switched parsers to use it. Added `defusedxml` to
`requirements.txt`.

## Hardening: ROS agent subprocess

`ros_agent.py` (scenario_runner + leaderboard) no longer uses `shell=True` for
`rosparam` / `start.sh`.
