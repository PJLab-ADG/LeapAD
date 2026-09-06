#!/usr/bin/env python3
"""Verify RunScript hardening blocks shell metacharacters."""
from __future__ import print_function

import os
import shlex
import subprocess
import sys
import tempfile

MARKER = "/tmp/pwned_leapad_rce_after_fix"


def fixed_runscript(script, base_path):
    try:
        script_components = shlex.split(script)
    except ValueError:
        return False
    if len(script_components) < 2:
        return False
    path = script_components[1]
    if not os.path.isfile(path):
        path = os.path.join(base_path, path)
    if not os.path.isfile(path):
        return False
    real_path = os.path.realpath(path)
    real_base = os.path.realpath(base_path)
    if not (real_path == real_base or real_path.startswith(real_base + os.sep)):
        return False
    argv = [script_components[0], real_path] + script_components[2:]
    subprocess.Popen(argv, shell=False, cwd=base_path).wait()
    return True


def main():
    if os.path.exists(MARKER):
        os.remove(MARKER)
    base = tempfile.mkdtemp(prefix="leapad_fix_")
    noop = os.path.join(base, "noop.py")
    with open(noop, "w") as f:
        f.write("print('noop')\n")
    payload = "python3 {} ; touch {}".format(noop, MARKER)
    fixed_runscript(payload, base)
    if os.path.exists(MARKER):
        print("FAIL: marker still created — fix ineffective")
        return 1
    print("PASS: shell metacharacters not executed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
