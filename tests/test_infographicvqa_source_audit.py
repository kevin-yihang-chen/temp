from __future__ import annotations

from beyond_entropy.infographicvqa_source_audit import (
    build_source_components,
    normalize_hostname,
)


def test_normalize_hostname_removes_only_network_noise() -> None:
    assert normalize_hostname("https://WWW.Example.COM:443/a?q=1") == "example.com"
    assert normalize_hostname("sub.example.com/path") == "sub.example.com"
    assert normalize_hostname("https://example.com./") == "example.com"
    assert normalize_hostname("") is None
    assert normalize_hostname("https:///") is None


def test_source_components_union_same_hostname_and_preserve_missing_host() -> None:
    source_by_rgb, members = build_source_components(
        {
            "rgb-a": {"site.example"},
            "rgb-b": {"site.example", "mirror.example"},
            "rgb-c": {"other.example"},
            "rgb-d": set(),
        }
    )
    assert source_by_rgb["rgb-a"] == source_by_rgb["rgb-b"]
    assert source_by_rgb["rgb-c"] != source_by_rgb["rgb-a"]
    assert source_by_rgb["rgb-d"] not in {
        source_by_rgb["rgb-a"],
        source_by_rgb["rgb-c"],
    }
    assert sorted(len(value) for value in members.values()) == [1, 1, 2]


def test_source_component_ids_are_order_invariant() -> None:
    first, _ = build_source_components(
        {"a": {"x"}, "b": {"x"}, "c": {"y"}}
    )
    second, _ = build_source_components(
        {"c": {"y"}, "b": {"x"}, "a": {"x"}}
    )
    assert first == second
