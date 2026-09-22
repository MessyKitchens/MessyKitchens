"""Regression tests for camera-ready object/token alignment."""

from pathlib import Path

from multi_object_decoder.object_ids import object_id_sort_key, parse_prefixed_object_id


def test_prefixed_object_ids_are_parsed_from_paths() -> None:
    assert parse_prefixed_object_id("obj_0007") == 7
    assert parse_prefixed_object_id(Path("tokens/obj_12.npz")) == 12
    assert parse_prefixed_object_id("img_cond_03.npy", prefix="img_cond") == 3
    assert parse_prefixed_object_id("object_3") is None


def test_object_ids_sort_numerically_not_lexicographically() -> None:
    values = ["obj_10.npz", "obj_2.npz", "obj_001.npz"]
    assert sorted(values, key=object_id_sort_key) == [
        "obj_001.npz",
        "obj_2.npz",
        "obj_10.npz",
    ]
