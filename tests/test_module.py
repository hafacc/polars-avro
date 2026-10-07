"""Benchmark avro reading and writing compared to native polars implementation."""

import tempfile
from collections.abc import Callable
from datetime import date, datetime
from functools import partial
from io import BytesIO

import jetliner
import polars as pl
import polars_fastavro
import pytest
from polars import DataFrame
from pytest_benchmark.fixture import BenchmarkFixture

import polars_avro

# naive so it can feed both the tz-aware and tz-naive datetime schemas
TIMESTAMP = datetime(2026, 1, 2, 3, 4, 5, 678000)  # noqa: DTZ001
DATE = TIMESTAMP.date()

# ------------------------- #
# Data generation functions #
# ------------------------- #


def create_narrow_frame(n: int) -> DataFrame:
    """Create a narrow frame with n rows and 2 columns."""
    return pl.from_dict(
        {"idx": [*range(n)], "name": [str(v) for v in range(n)]},
        schema={"idx": pl.Int32, "name": pl.String},
    )


def create_wide_frame(n: int) -> DataFrame:
    """Create a wide frame with n rows and 8 typed columns."""
    return pl.from_dict(
        {
            "i32_col": [*range(n)],
            "i64_col": [v * 1000 for v in range(n)],
            "f64_col": [v * 0.01 for v in range(n)],
            "f32_col": [v * 0.1 for v in range(n)],
            "bool_col": [v % 2 == 0 for v in range(n)],
            "str_col": [f"val_{v}" for v in range(n)],
            "binary_col": [f"bin_{v}".encode() for v in range(n)],
            "str2_col": [f"extra_{v}" for v in range(n)],
        },
        schema={
            "i32_col": pl.Int32,
            "i64_col": pl.Int64,
            "f64_col": pl.Float64,
            "f32_col": pl.Float32,
            "bool_col": pl.Boolean,
            "str_col": pl.String,
            "binary_col": pl.Binary,
            "str2_col": pl.String,
        },
    )


def create_nested_frame(n: int) -> DataFrame:
    """Create a nested frame with n rows, 8 columns, lists, and nested structs."""
    ratings = ["good", "mid", "bad"]
    return (
        pl.from_dict(
            {
                "idx": [*range(n)],
                "name": [f"name_{v}" for v in range(n)],
                "score": [v * 1.5 for v in range(n)],
                "active": [v % 2 == 0 for v in range(n)],
                "tags": [[f"tag_{t}" for t in range(v % 4)] for v in range(n)],
                "rating": [ratings[v % 3] for v in range(n)],
                "notes": [f"note_{v}" for v in range(n)],
            },
            schema={
                "idx": pl.Int32,
                "name": pl.String,
                "score": pl.Float64,
                "active": pl.Boolean,
                "tags": pl.List(pl.String),
                "rating": pl.String,
                "notes": pl.String,
            },
        )
        .lazy()
        .with_columns(  # type: ignore[reportUnknownMemberType]
            pl.struct(  # type: ignore[reportUnknownMemberType]
                pl.col("name"),
                pl.struct(pl.col("tags"), pl.col("score")),  # type: ignore[reportUnknownMemberType]
            ).alias("info"),
        )
        .collect()
    )


def create_mega_wide_frame(n: int) -> DataFrame:
    """Create a mega wide frame with n rows and 128 columns (16 copies of wide)."""
    base = create_wide_frame(n)
    return base.select(  # type: ignore[reportUnknownMemberType]
        *[pl.col("*").name.suffix(f"_{g}") for g in range(16)],
    )


def create_complex_frame(n: int) -> DataFrame:
    """Create a complex frame with n rows and 8 columns including nested types."""
    return (
        pl.from_dict(
            {
                "idx": [*range(n)],
                "name": [f"name_{v}" for v in range(n)],
                "score": [v * 1.5 for v in range(n)],
                "active": [v % 2 == 0 for v in range(n)],
                "tags": [[f"tag_{t}" for t in range(v % 5)] for v in range(n)],
                "key": [f"key_{v}" for v in range(n)],
                "value": [*range(n)],
                "created": [date.fromordinal(730120 + v % 365) for v in range(n)],
                "notes": [f"note_{v}" for v in range(n)],
            },
            schema={
                "idx": pl.Int32,
                "name": pl.String,
                "score": pl.Float64,
                "active": pl.Boolean,
                "tags": pl.List(pl.String),
                "key": pl.String,
                "value": pl.Int32,
                "created": pl.Date,
                "notes": pl.String,
            },
        )
        .lazy()
        .with_columns(  # type: ignore[reportUnknownMemberType]
            pl.struct(pl.col("key"), pl.col("value")).alias("metadata"),  # type: ignore[reportUnknownMemberType]
        )
        .select(
            pl.col("idx"),
            pl.col("name"),
            pl.col("score"),
            pl.col("active"),
            pl.col("tags"),
            pl.col("metadata"),
            pl.col("created"),
            pl.col("notes"),
        )
        .collect()
    )


# --------------------------- #
# Transitive round-trip tests #
# --------------------------- #


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(
            pl.from_dict({"col": [None, None]}, schema={"col": pl.Null}),
            id="nulls",
        ),
        pytest.param(
            pl.from_dict({"col": [True, False, None]}, schema={"col": pl.Boolean}),
            id="bools",
        ),
        pytest.param(
            pl.from_dict({"col": [-1, 0, 6, None]}, schema={"col": pl.Int32}),
            id="ints",
        ),
        pytest.param(
            pl.from_dict({"col": [-1, 0, 6, None]}, schema={"col": pl.Int64}),
            id="longs",
        ),
        pytest.param(
            pl.from_dict({"col": [DATE, None]}, schema={"col": pl.Date}),
            id="dates",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [TIMESTAMP, None]},
                schema={"col": pl.Datetime("ms", "UTC")},
            ),
            id="datetime-ms",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [TIMESTAMP, None]},
                schema={"col": pl.Datetime("us", "UTC")},
            ),
            id="datetime-us",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [TIMESTAMP, None]},
                schema={"col": pl.Datetime("ms")},
            ),
            id="datetime-ms-local",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [TIMESTAMP, None]},
                schema={"col": pl.Datetime("us")},
            ),
            id="datetime-us-local",
        ),
        pytest.param(
            pl.from_dict({"col": [-1.0, 0.0, 6.0, None]}, schema={"col": pl.Float32}),
            id="floats",
        ),
        pytest.param(
            pl.from_dict({"col": [-1.0, 0.0, 6.0, None]}, schema={"col": pl.Float64}),
            id="doubles",
        ),
        pytest.param(
            pl.from_dict(
                {"col": ["a", "b", None]}, schema={"col": pl.Enum(["a", "b"])}
            ),
            id="enum",
            marks=pytest.mark.xfail(
                reason="https://github.com/pola-rs/polars/issues/22273"
            ),
        ),
        pytest.param(
            pl.from_dict(
                {"col": [{"a": 5}, None, {"c": 8, "f": -10}]},
                schema={"col": pl.Map(pl.String, pl.Int32)},
            ),
            id="map",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [[{"a": 5}], None, [{"c": 8}, None]]},
                schema={"col": pl.List(pl.Map(pl.String, pl.Int32))},
            ),
            id="list of maps",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [{"map": {"a": [1.5]}}, {"map": None}]},
                schema={
                    "col": pl.Struct(
                        {"map": pl.Map(pl.String, pl.Array(pl.Float64, 1))}
                    )
                },
            ),
            id="struct of map",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [[None], []]},
                schema={"col": pl.List(pl.Null)},
            ),
            id="list of nulls",
        ),
        pytest.param(
            pl.from_dict({"col": [b"a", b"b", None]}, schema={"col": pl.Binary}),
            id="binary",
        ),
        pytest.param(
            pl.from_dict({"col": ["a", "b", None]}, schema={"col": pl.String}),
            id="string",
        ),
        pytest.param(
            pl.from_dict(
                {"col": [["a", None], ["b", "c"], ["d"], None]},
                schema={"col": pl.List(pl.String)},
            ),
            id="list-string",
        ),
        pytest.param(
            pl.from_dict(
                {
                    "col": [
                        {"s": "a", "i": 5},
                        {"s": "b", "i": None},
                        {"s": None, "i": 6},
                        {"s": None, "i": None},
                        None,
                    ]
                },
                schema={"col": pl.Struct({"s": pl.String, "i": pl.Int32})},
            ),
            id="struct",
        ),
        pytest.param(
            pl.from_dict(
                {
                    "col": [
                        [[{"s": "a", "i": 5}, None], [], None],
                        [[{"s": "b", "i": None}]],
                        [[{"s": None, "i": 6}], [{"s": None, "i": None}]],
                        None,
                    ]
                },
                schema={
                    "col": pl.List(pl.List(pl.Struct({"s": pl.String, "i": pl.Int32})))
                },
            ),
            id="nested",
        ),
        pytest.param(
            pl.from_dict(
                {
                    "struct": [[1, "a"], [None, "b"], [3, None]],
                    "single": [[1.0], [2.0], [3.0]],
                },
                schema={
                    "struct": pl.Struct({"a": pl.Int32, "b": pl.String}),
                    "single": pl.Struct({"x": pl.Float64}),
                },
            ),
            id="double struct",
        ),
        pytest.param(
            pl.from_dict(
                {
                    "one": ["a", "b", None],
                    "two": ["c", None, "d"],
                },
                schema={
                    "one": pl.Enum(["a", "b", "c"]),
                    "two": pl.Enum(["c", "d"]),
                },
            ),
            id="double enum",
            marks=pytest.mark.xfail(
                reason="https://github.com/pola-rs/polars/issues/22273"
            ),
        ),
    ],
)
def test_transitive(frame: pl.DataFrame):
    """Test that frames can be serialized and deserialized."""
    buff = BytesIO()
    polars_avro.write_avro(frame, buff)
    buff.seek(0)
    dup = polars_avro.read_avro(buff)

    assert frame.equals(dup)


# -------------------- #
# Non-contiguous tests #
# -------------------- #


@pytest.mark.parametrize(
    "write_func,read_func",
    [
        pytest.param(
            DataFrame.write_avro, pl.read_avro, id="polars", marks=pytest.mark.xfail
        ),
        pytest.param(polars_avro.write_avro, polars_avro.read_avro, id="polars_avro"),
    ],
)
def test_noncontiguous_chunks(
    write_func: Callable[[DataFrame, BytesIO], None],
    read_func: Callable[[BytesIO], pl.DataFrame],
) -> None:
    """Test that non contiguous arrays can be written and read."""
    frame = pl.concat(
        [
            pl.from_dict({"split": [*range(3)]}),
            pl.from_dict({"split": [*range(3, 6)]}),
        ],
        rechunk=False,
    ).with_columns(contig=pl.int_range(pl.len()))  # type: ignore
    buff = BytesIO()
    write_func(frame, buff)
    buff.seek(0)
    dup = read_func(buff)
    assert frame.equals(dup)


@pytest.mark.parametrize(
    "write_func,read_func",
    [
        pytest.param(DataFrame.write_avro, pl.read_avro, id="polars"),
        pytest.param(polars_avro.write_avro, polars_avro.read_avro, id="polars_avro"),
    ],
)
def test_noncontiguous_arrays(
    write_func: Callable[[DataFrame, BytesIO], None],
    read_func: Callable[[BytesIO], pl.DataFrame],
) -> None:
    """Test that non contiguous arrays can be written and read."""
    frame = pl.concat(
        [
            pl.from_dict({"split": [*range(3)]}),
            pl.from_dict({"split": [*range(3, 6)]}),
        ],
        rechunk=False,
    )
    buff = BytesIO()
    write_func(frame, buff)
    buff.seek(0)
    dup = read_func(buff)
    assert frame.equals(dup)


# ---------------- #
# Shape benchmarks #
# ---------------- #

SHAPES: list[tuple[str, Callable[[], DataFrame]]] = [
    ("narrow-small", partial(create_narrow_frame, 16)),
    ("narrow-medium", partial(create_narrow_frame, 1024)),
    ("narrow-long", partial(create_narrow_frame, 65536)),
    ("wide", partial(create_wide_frame, 1024)),
    ("large", partial(create_wide_frame, 1_048_576)),
    ("mega-wide", partial(create_mega_wide_frame, 1_048_576)),
]


@pytest.mark.parametrize(
    "frame_fn",
    [
        pytest.param(
            fn,
            id=name,
            marks=pytest.mark.benchmark(group=f"shape write {name}"),
        )
        for name, fn in SHAPES
    ],
)
@pytest.mark.parametrize(
    "write_func",
    [
        pytest.param(DataFrame.write_avro, id="polars"),
        pytest.param(polars_fastavro.write_avro, id="polars_fastavro"),
        pytest.param(polars_avro.write_avro, id="polars_avro"),
    ],
)
def test_shape_write(
    frame_fn: Callable[[], DataFrame],
    write_func: Callable[[DataFrame, BytesIO], None],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark writing across data shapes."""
    frame = frame_fn()

    def func() -> None:
        write_func(frame, BytesIO())

    benchmark(func)


@pytest.mark.parametrize(
    "frame_fn",
    [
        pytest.param(
            fn,
            id=name,
            marks=pytest.mark.benchmark(group=f"shape read {name}"),
        )
        for name, fn in SHAPES
    ],
)
@pytest.mark.parametrize(
    "read_func",
    [
        pytest.param(pl.read_avro, id="polars"),
        pytest.param(polars_fastavro.read_avro, id="polars_fastavro"),
        pytest.param(polars_avro.read_avro, id="polars_avro"),
    ],
)
def test_shape_read(
    frame_fn: Callable[[], DataFrame],
    read_func: Callable[[BytesIO], DataFrame],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark reading across data shapes."""
    frame = frame_fn()
    buff = BytesIO()
    polars_avro.write_avro(frame, buff)

    def func() -> None:
        buff.seek(0)
        read_func(buff)

    benchmark(func)


@pytest.mark.parametrize(
    "frame_fn",
    [
        pytest.param(
            fn,
            id=name,
            marks=pytest.mark.benchmark(group=f"shape read file {name}"),
        )
        for name, fn in SHAPES
    ],
)
@pytest.mark.parametrize(
    "read_func",
    [
        pytest.param(pl.read_avro, id="polars"),
        pytest.param(polars_fastavro.read_avro, id="polars_fastavro"),
        pytest.param(polars_avro.read_avro, id="polars_avro"),
        pytest.param(jetliner.read_avro, id="jetliner"),
    ],
)
def test_shape_read_file(
    frame_fn: Callable[[], DataFrame],
    read_func: Callable[[str], DataFrame],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark reading from file across data shapes."""
    frame = frame_fn()
    with tempfile.NamedTemporaryFile(suffix=".avro") as f:
        polars_avro.write_avro(frame, f.name)

        def func() -> None:
            read_func(f.name)

        benchmark(func)


# ------------------------------------------ #
# Nested & complex benchmarks (avro-only)    #
# ------------------------------------------ #

NESTED_SHAPES: list[tuple[str, Callable[[], DataFrame]]] = [
    ("nested", partial(create_nested_frame, 1024)),
    ("complex", partial(create_complex_frame, 1_048_576)),
]


@pytest.mark.parametrize(
    "frame_fn",
    [
        pytest.param(
            fn,
            id=name,
            marks=pytest.mark.benchmark(group=f"shape write {name}"),
        )
        for name, fn in NESTED_SHAPES
    ],
)
def test_nested_write(
    frame_fn: Callable[[], DataFrame],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark writing nested/complex data (polars_avro only)."""
    frame = frame_fn()

    def func() -> None:
        polars_avro.write_avro(frame, BytesIO())

    benchmark(func)


@pytest.mark.parametrize(
    "frame_fn",
    [
        pytest.param(
            fn,
            id=name,
            marks=pytest.mark.benchmark(group=f"shape read {name}"),
        )
        for name, fn in NESTED_SHAPES
    ],
)
def test_nested_read(
    frame_fn: Callable[[], DataFrame],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark reading nested/complex data (polars_avro only)."""
    frame = frame_fn()
    buff = BytesIO()
    polars_avro.write_avro(frame, buff)

    def func() -> None:
        buff.seek(0)
        polars_avro.read_avro(buff)

    benchmark(func)


@pytest.mark.parametrize(
    "frame_fn",
    [
        pytest.param(
            fn,
            id=name,
            marks=pytest.mark.benchmark(group=f"shape read file {name}"),
        )
        for name, fn in NESTED_SHAPES
    ],
)
@pytest.mark.parametrize(
    "read_func",
    [
        pytest.param(polars_avro.read_avro, id="polars_avro"),
        pytest.param(jetliner.read_avro, id="jetliner"),
    ],
)
def test_nested_read_file(
    frame_fn: Callable[[], DataFrame],
    read_func: Callable[[str], DataFrame],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark reading nested/complex data from file."""
    frame = frame_fn()
    with tempfile.NamedTemporaryFile(suffix=".avro") as f:
        polars_avro.write_avro(frame, f.name)

        def func() -> None:
            read_func(f.name)

        benchmark(func)


# --------------------- #
# Projection benchmarks #
# --------------------- #

PROJECTIONS: list[tuple[str, Callable[[], DataFrame], list[str]]] = [
    ("wide-2cols", partial(create_wide_frame, 1024), ["i32_col", "str_col"]),
    (
        "wide-5cols",
        partial(create_wide_frame, 1024),
        ["i32_col", "f64_col", "bool_col", "str_col", "binary_col"],
    ),
    (
        "mega-wide-8cols",
        partial(create_mega_wide_frame, 1_048_576),
        [
            "i32_col_0",
            "i64_col_0",
            "f64_col_0",
            "f32_col_0",
            "bool_col_0",
            "str_col_0",
            "binary_col_0",
            "str2_col_0",
        ],
    ),
]


@pytest.mark.parametrize(
    "frame_fn,columns",
    [
        pytest.param(
            fn,
            cols,
            id=name,
            marks=pytest.mark.benchmark(group=f"projection {name}"),
        )
        for name, fn, cols in PROJECTIONS
    ],
)
@pytest.mark.parametrize(
    "read_func",
    [
        pytest.param(pl.read_avro, id="polars"),
        pytest.param(polars_avro.read_avro, id="polars_avro"),
    ],
)
def test_projection(
    frame_fn: Callable[[], DataFrame],
    columns: list[str],
    read_func: Callable[..., DataFrame],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark reading with column projection."""
    frame = frame_fn()
    buff = BytesIO()
    polars_avro.write_avro(frame, buff)

    def func() -> None:
        buff.seek(0)
        read_func(buff, columns=columns)

    benchmark(func)


@pytest.mark.parametrize(
    "frame_fn,columns",
    [
        pytest.param(
            fn,
            cols,
            id=name,
            marks=pytest.mark.benchmark(group=f"projection file {name}"),
        )
        for name, fn, cols in PROJECTIONS
    ],
)
@pytest.mark.parametrize(
    "read_func",
    [
        pytest.param(pl.read_avro, id="polars"),
        pytest.param(polars_avro.read_avro, id="polars_avro"),
        pytest.param(jetliner.read_avro, id="jetliner"),
    ],
)
def test_projection_file(
    frame_fn: Callable[[], DataFrame],
    columns: list[str],
    read_func: Callable[..., DataFrame],
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark reading with column projection from file."""
    frame = frame_fn()
    with tempfile.NamedTemporaryFile(suffix=".avro") as f:
        polars_avro.write_avro(frame, f.name)

        def func() -> None:
            read_func(f.name, columns=columns)

        benchmark(func)


# ----------------------- #
# Read options benchmarks #
# ----------------------- #


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("utf8_view", [False, True])
@pytest.mark.benchmark(group="read options")
def test_read_options_bench(
    strict: bool,
    utf8_view: bool,
    benchmark: BenchmarkFixture,
) -> None:
    """Benchmark read options on medium frame."""
    frame = create_narrow_frame(1024)
    buff = BytesIO()
    polars_avro.write_avro(frame, buff)

    def func() -> None:
        buff.seek(0)
        polars_avro.read_avro(buff, strict=strict, utf8_view=utf8_view)  # type: ignore[reportArgumentType]

    benchmark(func)
