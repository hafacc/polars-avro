"""Test scan functionality."""

from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import fastavro
import fsspec  # type: ignore[reportMissingTypeStubs]
import polars as pl
import pytest

from polars_avro import read_avro, scan_avro, write_avro
from polars_avro._avro_rs import AvroSource


def test_scan_avro() -> None:
    """Test generic scan of files."""
    frame = scan_avro("resources/food.avro").with_row_index("row_index").collect()
    assert frame["row_index"].to_list() == [*range(27)]

    frame = (
        scan_avro("resources/food.avro")
        .with_row_index("row_index")
        .filter(pl.col("category") == pl.lit("vegetables"))  # type: ignore
        .collect()
    )
    assert frame["row_index"].to_list() == [0, 6, 11, 13, 14, 20, 25]

    frame = (
        scan_avro("resources/food.avro")
        .with_row_index("foo", 10)
        .filter(pl.col("category") == pl.lit("vegetables"))  # type: ignore
        .collect()
    )
    assert frame["foo"].to_list() == [10, 16, 21, 23, 24, 30, 35]


def test_projection_pushdown_avro() -> None:
    """Test that projection is pushed down to scan."""
    file_path = "resources/food.avro"
    lazy = scan_avro(file_path).select(pl.col.calories)  # type: ignore

    explain = lazy.explain()

    assert "simple π" not in explain
    assert "PROJECT 1/4 COLUMNS" in explain

    normal = lazy.collect()
    unoptimized = lazy.collect(optimizations=pl.QueryOptFlags.none())
    assert normal.equals(unoptimized)


def test_predicate_pushdown_avro() -> None:
    """Test that predicate is pushed down to scan."""
    file_path = "resources/food.avro"
    thresh = 80
    lazy = scan_avro(file_path).filter(pl.col("calories") > thresh)  # type: ignore

    explain = lazy.explain()

    assert "FILTER" not in explain
    # polars renders the expression itself differently between versions
    (selection,) = [
        line for line in explain.splitlines() if line.strip().startswith("SELECTION:")
    ]
    assert "calories" in selection
    assert str(thresh) in selection

    normal = lazy.collect()
    unoptimized = lazy.collect(optimizations=pl.QueryOptFlags.none())
    assert normal.equals(unoptimized)


def test_many_files() -> None:
    """Test that scan works with many files."""
    buff = BytesIO()
    frame = pl.from_dict({"x": [5, 12, 14]})
    write_avro(frame, buff)

    buffs = [BytesIO(buff.getvalue()) for _ in range(1023)]
    res = scan_avro(buffs).collect()
    reference = pl.from_dict({"x": [5, 12, 14] * 1023})
    assert res.equals(reference)


def test_glob_n_rows() -> None:
    """Test that globbing and n_rows work."""
    file_path = "resources/*.avro"
    frame = scan_avro(file_path).limit(28).collect()

    # 27 rows from food.avro and 1 from grains.avro
    assert frame.shape == (28, 4)

    # take first and last rows
    assert frame[[0, 27]].to_dict(as_series=False) == {
        "category": ["vegetables", "rice"],
        "calories": [45, 9],
        "fats_g": [0.5, 0.0],
        "sugars_g": [2, 0.3],
    }


def test_scan_nrows_empty() -> None:
    """Test that scan doesn't panic with n_rows set to 0."""
    file_path = "resources/food.avro"
    frame = scan_avro(file_path).head(0).collect()
    reference = read_avro(file_path).head(0)
    assert frame.equals(reference)


def test_scan_filter_empty() -> None:
    """Test that scan doesn't panic when filter removes all rows."""
    file_path = "resources/food.avro"
    frame = scan_avro(file_path).filter(pl.col("category") == "empty").collect()  # type: ignore
    reference = read_avro(file_path).filter(pl.col("category") == "empty")  # type: ignore
    assert frame.equals(reference)


def test_directory() -> None:
    """Test scan on directory."""
    frame = scan_avro("resources").collect()
    assert frame.shape == (30, 4)


def test_avro_list_arg() -> None:
    """Test that scan works when passing a list."""
    first = "resources/food.avro"
    second = "resources/grains.avro"

    frame = scan_avro([first, second]).collect()
    assert frame.shape == (30, 4)
    assert frame.row(-1) == ("corn", 99, 0.1, 10.4)
    assert frame.row(0) == ("vegetables", 45, 0.5, 2)


def test_mixed_source_order_preserved(tmp_path: Path) -> None:
    """Rows follow argument order even when mixing local paths and buffers."""
    path = tmp_path / "later.avro"
    write_avro(pl.DataFrame({"x": [3, 4]}), path)

    def buf() -> BytesIO:
        handle = BytesIO()
        write_avro(pl.DataFrame({"x": [1, 2]}), handle)
        handle.seek(0)
        return handle

    # buffer before path: rows must not jump behind the local file
    assert scan_avro([buf(), str(path)]).collect()["x"].to_list() == [1, 2, 3, 4]
    # and the reverse order reverses the result
    assert scan_avro([str(path), buf()]).collect()["x"].to_list() == [3, 4, 1, 2]


def test_glob_single_scan() -> None:
    """Test that globbing works with a single file."""
    file_path = "resources/food*.avro"
    frame = scan_avro(file_path)

    explain = frame.explain()

    assert explain.count("SCAN") == 1
    assert "UNION" not in explain


def test_source_exception_type_preserved() -> None:
    """A python exception from a source keeps its type across the rust boundary.

    Reading through ``AvroSource`` directly avoids polars' io-plugin wrapper, so
    the original exception (type, message, traceback) is observable — a
    ``KeyboardInterrupt`` from a slow read is no longer flattened to a string.
    """

    class Interrupted(Exception):
        pass

    @contextmanager
    def factory() -> Generator[BinaryIO, None, None]:
        class Reader:
            def read(self, _size: int = -1) -> bytes:
                raise Interrupted("connection dropped")

            def seek(self, pos: int, _whence: int = 0) -> int:
                return pos

            def tell(self) -> int:
                return 0

        yield Reader()  # type: ignore[misc]

    source = AvroSource([factory], False, False)
    with pytest.raises(Interrupted, match="connection dropped"):
        source.schema()


def test_glob_no_match_errors() -> None:
    """A glob matching no files raises instead of silently dropping data."""
    with pytest.raises(FileNotFoundError, match="no files matched"):
        scan_avro("resources/nomatch*.avro")

    # even alongside a good source, a typo'd pattern must not silently vanish
    with pytest.raises(FileNotFoundError, match="no files matched"):
        scan_avro(["resources/food.avro", "resources/nomatch*.avro"])


def test_empty_directory_errors(tmp_path: Path) -> None:
    """Scanning an empty directory raises rather than yielding no data."""
    with pytest.raises(FileNotFoundError, match="no files found in directory"):
        scan_avro(str(tmp_path))


def test_batch_size_must_be_positive() -> None:
    """A batch size that could never yield rows is rejected."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        scan_avro("resources/food.avro", batch_size=0)


def test_utf8_view_schema_matches_data() -> None:
    """The declared schema follows `utf8_view`, which the streaming engine checks."""
    buff = BytesIO()
    fastavro.writer(  # type: ignore
        buff,
        {
            "type": "record",
            "name": "ids",
            "fields": [
                {"name": "id", "type": {"type": "string", "logicalType": "uuid"}}
            ],
        },
        [{"id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"}],
    )
    for utf8_view, dtype in [(False, pl.Binary), (True, pl.String)]:
        buff.seek(0)
        lazy = scan_avro(buff, utf8_view=utf8_view)
        assert lazy.collect_schema() == {"id": dtype}
        assert lazy.collect(engine="streaming").schema == {"id": dtype}


@pytest.mark.xfail(
    strict=True,
    reason="arrow-avro's projection drops the definition of a type it still uses",
)
def test_projection_of_shared_named_type() -> None:
    """A column can be selected without the column that defines its type."""
    point = {
        "type": "record",
        "name": "Point",
        "fields": [{"name": "x", "type": "int"}],
    }
    buff = BytesIO()
    fastavro.writer(  # type: ignore
        buff,
        {
            "type": "record",
            "name": "base",
            "fields": [
                {"name": "first", "type": point},
                {"name": "second", "type": "Point"},
            ],
        },
        [{"first": {"x": 1}, "second": {"x": 2}}],
    )
    buff.seek(0)
    expected = pl.from_dict(
        {"second": [{"x": 2}]}, schema={"second": pl.Struct({"x": pl.Int32})}
    )
    assert read_avro(buff, columns=["second"]).equals(expected)


def test_scan_in_memory() -> None:
    """Test that scan works for in memory buffers."""
    frame = pl.from_dict({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    buff = BytesIO()
    write_avro(frame, buff)

    buff.seek(0)
    scanned = scan_avro(buff).collect()
    assert frame.equals(scanned)

    buff.seek(0)
    scanned = scan_avro(buff).slice(1, 2).collect()
    assert frame.slice(1, 2).equals(scanned)

    buff.seek(0)
    scanned = scan_avro(buff).slice(-1, 1).collect()
    assert frame.slice(-1, 1).equals(scanned)

    other = BytesIO(buff.getvalue())

    buff.seek(0)
    scanned = scan_avro([buff, other]).collect()
    assert pl.concat([frame, frame]).equals(scanned)

    buff.seek(0)
    other.seek(0)
    scanned = scan_avro([buff, other]).slice(1, 3).collect()
    assert pl.concat([frame, frame]).slice(1, 3).equals(scanned)

    buff.seek(0)
    other.seek(0)
    scanned = scan_avro([buff, other]).slice(-4, 3).collect()
    assert pl.concat([frame, frame]).slice(-4, 3).equals(scanned)


def test_read_map_type() -> None:
    """Test that we can read a map type."""
    buff = BytesIO()
    values = [{"map": {"a": 5}}, {"map": None}, {"map": {"c": 8, "f": -10}}]
    fastavro.writer(  # type: ignore
        buff,
        {
            "type": "record",
            "name": "map_test",
            "fields": [
                {"name": "map", "type": ["null", {"type": "map", "values": "int"}]}
            ],
        },
        values,
    )
    buff.seek(0)
    res = read_avro(buff)
    expected = pl.from_dict(
        {"map": [{"a": 5}, None, {"c": 8, "f": -10}]},
        schema={"map": pl.Map(pl.String, pl.Int32)},
    )
    assert res.equals(expected)


def test_read_options() -> None:
    """Test read works with options."""
    frame = read_avro(
        "resources/food.avro", row_index_name="row_index", columns=[1], n_rows=11
    )
    assert frame.shape == (11, 2)
    assert frame["row_index"].to_list() == [*range(11)]


def test_projection_preserves_logical_types() -> None:
    """Projection and strict mode must keep logical types, not raw primitives."""
    frame = pl.DataFrame(
        {
            "ts": pl.Series([0], dtype=pl.Datetime("us")),
            "d": pl.Series([18262], dtype=pl.Int32).cast(pl.Date),
            "x": pl.Series([42], dtype=pl.Int64),
        }
    )
    buff = BytesIO()
    write_avro(frame, buff)

    buff.seek(0)
    projected = scan_avro(buff).select("ts", "d").collect()  # type: ignore
    assert projected.schema == {"ts": pl.Datetime("us"), "d": pl.Date}
    assert projected["ts"].to_list() == frame["ts"].to_list()

    buff.seek(0)
    strict = scan_avro(buff, strict=True).collect()
    assert strict.schema == frame.schema


def test_projection_different_types_errors() -> None:
    """Two sources with a shared column of different types error under projection."""
    one = BytesIO()
    write_avro(pl.DataFrame({"x": pl.Series([1], dtype=pl.Int64)}), one)
    two = BytesIO()
    write_avro(pl.DataFrame({"x": ["a"]}), two)
    one.seek(0)
    two.seek(0)
    with pytest.raises(Exception):  # noqa: B017
        scan_avro([one, two]).select("x").collect()  # type: ignore


def test_filename_in_err() -> None:
    """Test that invalid filename is reported in error."""
    lazy = scan_avro("does not exist")
    with pytest.raises(Exception, match="does not exist"):
        lazy.collect()


def test_empty_sources() -> None:
    """Test that empty sources raises an error."""
    lazy = scan_avro([])
    with pytest.raises(Exception, match="must scan at least one source"):
        lazy.collect()


def test_cloud_scan() -> None:
    """Test scanning a cloud URL through fsspec (using its in-process memory fs)."""
    frame = pl.from_dict({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    buff = BytesIO()
    write_avro(frame, buff)
    with fsspec.open("memory://cloud_test.avro", "wb") as handle:  # type: ignore[reportUnknownMemberType]
        handle.write(buff.getvalue())  # type: ignore[reportUnknownMemberType]

    # memory:// is a cloud scheme, so scan_avro routes it through fsspec
    assert read_avro("memory://cloud_test.avro").equals(frame)

    # scan, re-collect, and a projection (exercises the per-scan fsspec re-open)
    lazy = scan_avro("memory://cloud_test.avro")
    assert lazy.collect().equals(frame)
    assert lazy.collect().equals(frame)
    assert read_avro(
        "memory://cloud_test.avro", columns=["y"]
    ).to_series().to_list() == [
        "a",
        "b",
        "c",
    ]


class SentinelError(AssertionError):
    """A sentinel error for raising."""
