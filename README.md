# polars-avro

[![build](https://github.com/hafacc/polars-avro/actions/workflows/build.yml/badge.svg)](https://github.com/hafacc/polars-avro/actions/workflows/build.yml)
[![pypi](https://img.shields.io/pypi/v/polars-avro)](https://pypi.org/project/polars-avro/)
[![docs](https://img.shields.io/badge/api-docs-blue)](https://hafa.cc/polars-avro)

A polars io plugin for reading and writing
[Apache Avro](https://avro.apache.org/) files, built on
[arrow-avro](https://crates.io/crates/arrow-avro). It provides scan support
with predicate pushdown, map type reading, and continued avro support as polars
deprecates its built-in implementation.

## Python Usage

```py
from polars_avro import scan_avro, read_avro, write_avro

lazy = scan_avro(path)
frame = read_avro(path)
write_avro([frame], path)
```

Both `scan_avro` and `read_avro` accept cloud and other URLs (`s3://`, `gs://`,
`az://`, `http(s)://`, ...), read through
[fsspec](https://filesystem-spec.readthedocs.io/) — install the relevant backend
(e.g. `s3fs`) and pass `storage_options` (forwarded to `fsspec.open`) for
credentials:

```py
lazy = scan_avro("s3://bucket/data.avro", storage_options={"anon": True})
frame = read_avro("s3://bucket/data.avro", storage_options={"anon": True})
```

## Rust Usage

There are two main exports: [`Reader`] for iterating arrow `RecordBatch`es from
avro sources, and [`Writer`] for writing `RecordBatch`es to an avro file.

```rs
use polars_avro::{FullReadOptions, Reader, Writer};
use std::fs::File;

// `Reader` yields arrow `RecordBatch`es from one or more avro sources
let mut reader = Reader::try_new(
    [File::open("data.avro")],
    FullReadOptions::default(),
).unwrap();

// copy them into a new file; `Writer` needs a schema up front, so take it
// from the first batch
let first = reader.next().unwrap().unwrap();
let mut writer = Writer::try_new(
    File::create("copy.avro").unwrap(),
    first.schema(),
    None,
).unwrap();
writer.write(&first).unwrap();
for batch in reader {
    writer.write(&batch.unwrap()).unwrap();
}
writer.finish().unwrap();
```

> ℹ️ Avro supports writing with file compression schemes. In rust these need
> to be enabled via feature flags: `deflate`, `snappy`, `bzip2`, `xz`, `zstd`.
> Decompression is handled automatically.

## Idiosyncrasies

Avro and Arrow don't align fully, and polars only supports a subset of arrow.
Some types require casting before writing, and some avro types map to different
polars types than you might expect when reading.

### Writing

The following polars types **error** when writing and must be cast first:

| Polars Type    | Cast To                    |
| -------------- | -------------------------- |
| large `UInt64` | Wrap to `Int64`            |
| `Categorical`  | `Int32` or `String`        |
| `Enum`         | `Int32` or `String`        |

Times will get truncated to micro seconds.

Compression is supported via feature flags: `deflate`, `snappy`, `bzip2`, `xz`,
`zstd`.

### Reading

**`utf8_view` behavior** — the `utf8_view` option (default `false`) changes how
certain types are read:

| Type             | `utf8_view=false` (default) | `utf8_view=true`               |
| ---------------- | --------------------------- | ------------------------------ |
| UUID             | binary (16 bytes)           | formatted string               |
| nullable strings | preserves nulls             | replaces null with `""` (lossy)|

Since polars tends to work with string views internally, `utf8_view=true` is
likely faster if you don't mind losing null string distinctions.

**Type mappings of note:**

| Avro Type                          | Polars Type                                   |
| ---------------------------------- | --------------------------------------------- |
| Enum                               | Categorical (not Enum)                        |
| Map                                | List of Struct {key, value}                   |
| BigDecimal                         | Binary                                        |
| Duration                           | unsupported (errors)                          |
| Date                               | Date (days since epoch)                       |
| TimeMillis, TimeMicros             | Time (nanoseconds)                            |
| TimestampMillis/Micros/Nanos       | Datetime with matching precision and UTC tz   |
| LocalTimestampMillis/Micros/Nanos  | Datetime with matching precision and no tz    |

**Constraints:** the root avro schema must be a Record, and all files in a
multi-file read must share the same schema.

## Benchmarks

Python reports median (file reads, in-memory writes). Rust reports mean.
`native` = polars built-in avro. Ratio relative to native; **bold** = fastest.
Complex rows use nested/struct types.

| Benchmark                      |              native |         polars-avro |        jetliner |
| ------------------------------ | ------------------: | ------------------: | --------------: |
| python read 1K × 2             |   **64 µs** (1.00x) |       99 µs (1.54x) |  180 µs (2.79x) |
| python read 64K × 2            |      2.7 ms (1.00x) |  **2.1 ms** (0.78x) |  2.8 ms (1.04x) |
| python read 1K × 8             |  **183 µs** (1.00x) |      242 µs (1.32x) |  337 µs (1.84x) |
| python read 1M × 8             |      159 ms (1.00x) |  **114 ms** (0.72x) |  145 ms (0.91x) |
| python read 1M × 128           |       2.6 s (1.00x) |   **1.8 s** (0.69x) |   2.8 s (1.09x) |
| python read complex 1K × 8     |                   — |          **449 µs** |          592 µs |
| python read complex 1M × 8     |                   — |          **181 ms** |          260 ms |
| python read proj 1M × 128 → 8  |       1.6 s (1.00x) |   **1.2 s** (0.75x) |   1.2 s (0.77x) |
| python read proj 1K × 8 → 2    |  **133 µs** (1.00x) |      297 µs (2.24x) |  264 µs (1.99x) |
| python write 1K × 2            |       42 µs (1.00x) |   **30 µs** (0.72x) |               — |
| python write 64K × 2           |      1.5 ms (1.00x) |  **1.1 ms** (0.71x) |               — |
| python write 1K × 8            |      143 µs (1.00x) |  **114 µs** (0.80x) |               — |
| python write 1M × 8            |   **87 ms** (1.00x) |       93 ms (1.07x) |               — |
| python write 1M × 128          |   **1.5 s** (1.00x) |       2.2 s (1.48x) |               — |
| rust read 1K × 2               |       42 µs (1.00x) |   **34 µs** (0.80x) |               — |
| rust read 1M × 128             |       2.8 s (1.00x) |   **2.0 s** (0.69x) |               — |
| rust read proj 1M × 128 → 8    |       1.3 s (1.00x) |   **1.2 s** (0.87x) |               — |
| rust read proj 1K × 8 → 2      |  **109 µs** (1.00x) |      116 µs (1.06x) |               — |
| rust write 1K × 2              |       42 µs (1.00x) |   **22 µs** (0.53x) |               — |
| rust write 64K × 2             |      1.5 ms (1.00x) |  **1.0 ms** (0.67x) |               — |
| rust write 1K × 8              |      135 µs (1.00x) |   **93 µs** (0.69x) |               — |
| rust write 1M × 8              |       97 ms (1.00x) |   **89 ms** (0.92x) |               — |
| rust write 1M × 128            |       1.6 s (1.00x) |   **1.4 s** (0.88x) |               — |

## Development

### Rust

Standard `cargo` commands will build and test the rust library.

### Python

The python library is built with uv and maturin. `uv sync` compiles the rust
extension once, after which you can use and test the library from python.

You may need to recompile the python bindings with `uv run maturin develop`.

### Testing

```sh
cargo fmt --check
cargo clippy --all-features --tests -- -D warnings
cargo test
uv run ruff format --check
uv run ruff check
uv run pyright
uv run pytest
```

### Benchmarking

Benchmarks must run against an **optimized** build. `maturin develop` (used for
normal testing) compiles the extension unoptimized, which makes the Python
benchmarks meaningless (~20x slower). Build release first:

```sh
cargo +nightly bench
uv run maturin develop --release
uv run pytest --benchmark-only
```

### Releasing

Releases are automated. Trigger the
[`release`](.github/workflows/release.yml) workflow from the Actions tab (or
`gh workflow run release.yml`), choosing the version bump (`patch`, `minor`, or
`major`). It gates on the full test suite, builds wheels and an sdist, commits
and tags the version bump, and publishes to PyPI via OIDC trusted publishing.
