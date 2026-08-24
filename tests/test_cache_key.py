from rgmt.data.cache_key import SCHEMA_VERSION, file_sha256, params_key

def test_file_sha256_deterministic(tmp_path):
    p = tmp_path / "a.bin"; p.write_bytes(b"hello world")
    assert file_sha256(p) == file_sha256(p)
    q = tmp_path / "b.bin"; q.write_bytes(b"hello world!")
    assert file_sha256(p) != file_sha256(q)

def test_params_key_changes_on_any_input():
    base = dict(source_hash="s", urdf_hash="u", physics_fps=60, src_fps=30,
                keypoint_links=["a", "b"], schema_version=SCHEMA_VERSION)
    k0 = params_key(**base)
    assert k0 == params_key(**base)  # stable
    for field, val in [("source_hash", "s2"), ("urdf_hash", "u2"),
                       ("physics_fps", 50), ("src_fps", 25),
                       ("keypoint_links", ["a"]), ("schema_version", 999)]:
        assert params_key(**{**base, field: val}) != k0
