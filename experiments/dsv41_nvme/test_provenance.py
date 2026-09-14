"""Contract test: the extracted implementations must keep their recorded bytes.

`source-provenance.json` records the sha256 of each unchanged implementation.
The packed K2-K8 records, cancellation, alignment and checksum behaviour under
review are exactly those bytes, so silent drift here invalidates the review.
"""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


class ProvenanceTests(unittest.TestCase):
    def test_implementation_bytes_match_recorded_provenance(self):
        recorded = json.loads((ROOT / "source-provenance.json").read_text())["sources"]
        self.assertTrue(recorded, "provenance file must pin the extracted implementations")
        for name, entry in recorded.items():
            source = ROOT / name
            self.assertTrue(source.is_file(), f"{name} is missing")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"], f"{name} drifted from its recorded extraction")
            self.assertTrue(entry["extracted_without_code_changes"])

    def test_pinned_implementations_are_not_imported_as_an_installed_backend(self):
        # The components stay an isolated experiment: no production module may
        # import them, and no plugin entry point may register them.
        text = ROOT.parent.parent.joinpath("pyproject.toml").read_text()
        for name in ("expert_store", "expert_cache", "async_store", "engram_rows"):
            self.assertNotIn(name, text)


if __name__ == "__main__":
    unittest.main()
