import hashlib
import json

from academic_assistant.utils.manifest import (
    fingerprint,
    load_manifest,
    sha256_file,
    utc_now_iso,
    write_manifest_atomic,
)


def test_sha256_and_fingerprint(tmp_path):
    f = tmp_path / "a.md"
    f.write_bytes(b"hello")
    assert sha256_file(f) == hashlib.sha256(b"hello").hexdigest()
    fp = fingerprint(f)
    assert fp.sha256 == sha256_file(f)
    assert fp.size == 5


def test_load_manifest_missing_returns_default(tmp_path):
    default = {"files": {}}
    assert load_manifest(tmp_path / "nope.json", default) is default


def test_write_then_load_roundtrip_without_leftovers(tmp_path):
    path = tmp_path / "sub" / "manifest.json"
    manifest = {"files": {"summaries/a.md": {"sha256": "abc"}}, "manifest_version": 1}
    write_manifest_atomic(path, manifest)
    assert load_manifest(path, default={}) == manifest
    assert json.loads(path.read_text()) == manifest
    assert [p.name for p in path.parent.iterdir()] == ["manifest.json"]


def test_utc_now_iso_is_utc():
    assert utc_now_iso().endswith("+00:00")
