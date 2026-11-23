from __future__ import annotations

import pytest

from tests._sharding import owns_nodeid, parse_shard


def test_parse_shard_reads_index_and_total() -> None:
    assert parse_shard("1/2") == (1, 2)
    assert parse_shard("3/7") == (3, 7)


@pytest.mark.parametrize("spec", ["0/2", "3/2", "1/0", "banana", "", "1/2/3"])
def test_parse_shard_rejects_invalid_specs(spec: str) -> None:
    with pytest.raises(pytest.UsageError):
        parse_shard(spec)


@pytest.mark.parametrize("total", [2, 3, 4])
def test_every_node_id_belongs_to_exactly_one_shard(total: int) -> None:
    nodeids = [f"tests/pkg/test_module_{n}.py::test_case_{n}" for n in range(500)]

    owners = [sum(owns_nodeid(nodeid, index, total) for index in range(1, total + 1)) for nodeid in nodeids]

    assert owners == [1] * len(nodeids)
    for index in range(1, total + 1):
        assert any(owns_nodeid(nodeid, index, total) for nodeid in nodeids)


def test_two_shards_split_a_realistic_collection_in_half() -> None:
    nodeids = [f"tests/cli/test_app_{n}.py::test_flow_{n}" for n in range(400)]

    first = sum(owns_nodeid(nodeid, 1, 2) for nodeid in nodeids)

    assert 160 <= first <= 240
