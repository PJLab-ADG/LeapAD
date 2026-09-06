#!/usr/bin/env python
"""Safe XML helpers — prefer defusedxml to mitigate XXE when loading scenarios/routes."""

from __future__ import print_function

try:
    import defusedxml.ElementTree as ET
except ImportError:  # pragma: no cover - fallback for environments without defusedxml
    import xml.etree.ElementTree as _ET

    class _RejectDTDParser(_ET.XMLParser):
        def __init__(self, *args, **kwargs):
            kwargs = dict(kwargs)
            # Python 3 XMLParser does not expose a simple forbid_dtd flag on all versions;
            # reject DTD/ENTITY payloads before parsing instead.
            super(_RejectDTDParser, self).__init__(*args, **kwargs)

    def _forbid_entities(source):
        if hasattr(source, "read"):
            data = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
        else:
            with open(source, "rb") as fh:
                data = fh.read()
        upper = data.upper()
        if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
            raise ValueError("XML with DOCTYPE/ENTITY is not allowed (XXE mitigation)")
        return data

    class ET(object):
        @staticmethod
        def parse(source, parser=None):
            data = _forbid_entities(source)
            return _ET.ElementTree(_ET.fromstring(data))

        @staticmethod
        def fromstring(text):
            if isinstance(text, bytes):
                raw = text
            else:
                raw = text.encode("utf-8")
            upper = raw.upper()
            if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
                raise ValueError("XML with DOCTYPE/ENTITY is not allowed (XXE mitigation)")
            return _ET.fromstring(raw)
