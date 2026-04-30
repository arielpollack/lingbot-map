"""Round-trip tests for the .splat binary encoder."""

import struct

import numpy as np

from poc.worker.splat_format import (
    SPLAT_BYTES_PER_GAUSSIAN,
    decode_splat_file,
    encode_splat_file,
)


def _make_inputs(n: int, *, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return {
        "means": rng.normal(size=(n, 3)).astype(np.float32),
        "quats_wxyz": quats,
        "scales": np.abs(rng.normal(size=(n, 3))).astype(np.float32) + 0.01,
        "opacities": rng.uniform(0.1, 0.9, size=(n,)).astype(np.float32),
        "rgb": rng.integers(0, 256, size=(n, 3), dtype=np.uint8),
    }


def test_encode_writes_expected_byte_count(tmp_path):
    inputs = _make_inputs(7)
    out = tmp_path / "out.splat"
    info = encode_splat_file(**inputs, output_path=str(out))
    assert out.stat().st_size == 7 * SPLAT_BYTES_PER_GAUSSIAN
    assert info["bytes"] == 7 * SPLAT_BYTES_PER_GAUSSIAN
    assert info["gaussians"] == 7
    assert info["format"] == "splat"
    assert info["sorted"] is False


def test_round_trip_recovers_position_and_scale_exactly(tmp_path):
    inputs = _make_inputs(50)
    out = tmp_path / "out.splat"
    encode_splat_file(**inputs, output_path=str(out))
    decoded = decode_splat_file(str(out))

    np.testing.assert_array_equal(decoded["means"], inputs["means"])
    np.testing.assert_array_equal(decoded["scales"], inputs["scales"])
    np.testing.assert_array_equal(decoded["rgb"], inputs["rgb"])


def test_round_trip_quat_within_quantization_error(tmp_path):
    inputs = _make_inputs(50)
    out = tmp_path / "out.splat"
    encode_splat_file(**inputs, output_path=str(out))
    decoded = decode_splat_file(str(out))

    # uint8 quaternion encoding is q*128+128, so worst-case error per channel
    # is 1/128. Real quats are unit-norm, so per-component magnitude ≤ 1.
    diff = decoded["quats_wxyz"] - inputs["quats_wxyz"]
    assert np.all(np.abs(diff) < 1.5 / 128.0), float(np.abs(diff).max())


def test_round_trip_opacity_within_quantization_error(tmp_path):
    inputs = _make_inputs(50)
    out = tmp_path / "out.splat"
    encode_splat_file(**inputs, output_path=str(out))
    decoded = decode_splat_file(str(out))

    diff = decoded["opacities"] - inputs["opacities"]
    assert np.all(np.abs(diff) < 1.5 / 255.0)


def test_layout_matches_spec_offsets(tmp_path):
    inputs = _make_inputs(1)
    inputs["means"] = np.array([[1.5, -2.25, 3.75]], dtype=np.float32)
    inputs["scales"] = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    inputs["rgb"] = np.array([[255, 128, 0]], dtype=np.uint8)
    inputs["opacities"] = np.array([0.5], dtype=np.float32)
    # Identity quaternion (w=1, x=y=z=0).
    inputs["quats_wxyz"] = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    out = tmp_path / "out.splat"
    encode_splat_file(**inputs, output_path=str(out))
    raw = out.read_bytes()
    assert len(raw) == SPLAT_BYTES_PER_GAUSSIAN

    pos = struct.unpack_from("<3f", raw, 0)
    scale = struct.unpack_from("<3f", raw, 12)
    r, g, b, a = struct.unpack_from("<4B", raw, 24)
    rot = struct.unpack_from("<4B", raw, 28)

    # Position values are exact powers of 2 fractions → represent exactly in f32.
    assert pos == (1.5, -2.25, 3.75)
    # Scales 0.1/0.2/0.3 don't represent exactly in float32; compare with tolerance.
    np.testing.assert_allclose(scale, (0.1, 0.2, 0.3), rtol=0, atol=1e-6)
    assert (r, g, b) == (255, 128, 0)
    # opacity 0.5 → alpha ≈ 128
    assert abs(a - 128) <= 1
    # Identity quat → (1, 0, 0, 0) → encoded (256→255, 128, 128, 128)
    assert rot == (255, 128, 128, 128)


def test_sort_by_distance_orders_by_distance(tmp_path):
    # Three Gaussians at increasing distance from origin.
    means = np.array(
        [[10.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1))
    scales = np.ones((3, 3), dtype=np.float32) * 0.1
    opacities = np.ones(3, dtype=np.float32) * 0.5
    rgb = np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=np.uint8)

    out = tmp_path / "out.splat"
    info = encode_splat_file(
        means=means,
        quats_wxyz=quats,
        scales=scales,
        opacities=opacities,
        rgb=rgb,
        output_path=str(out),
        sort_by_distance_to=(0.0, 0.0, 0.0),
    )
    assert info["sorted"] is True

    decoded = decode_splat_file(str(out))
    # After distance sort: nearest (means[1]) → middle (means[2]) → furthest (means[0])
    np.testing.assert_array_equal(
        decoded["means"],
        np.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(decoded["rgb"], np.array([[20, 20, 20], [30, 30, 30], [10, 10, 10]], dtype=np.uint8))


def test_zero_length_quat_encodes_as_identity(tmp_path):
    inputs = _make_inputs(2)
    inputs["quats_wxyz"][0] = 0.0  # bad quaternion
    out = tmp_path / "out.splat"
    encode_splat_file(**inputs, output_path=str(out))
    decoded = decode_splat_file(str(out))

    # Should fall back to identity (w≈1, x=y=z≈0).
    assert decoded["quats_wxyz"][0, 0] >= 0.99
    assert abs(decoded["quats_wxyz"][0, 1]) < 0.01
    assert abs(decoded["quats_wxyz"][0, 2]) < 0.01
    assert abs(decoded["quats_wxyz"][0, 3]) < 0.01


def test_shape_mismatch_raises(tmp_path):
    inputs = _make_inputs(5)
    inputs["rgb"] = inputs["rgb"][:3]  # wrong length
    out = tmp_path / "out.splat"
    try:
        encode_splat_file(**inputs, output_path=str(out))
    except ValueError as e:
        assert "Inconsistent shapes" in str(e)
    else:
        raise AssertionError("expected ValueError on shape mismatch")
