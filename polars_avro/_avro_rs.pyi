from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import BinaryIO

from polars._typing import ArrowStreamExportable

class Codec:
    """A compression codec to use when writing Avro files.

    Pass ``None`` instead of a codec for no compression (the avro ``null``
    codec).
    """

    Bzip2: Codec
    Deflate: Codec
    Snappy: Codec
    Xz: Codec
    Zstandard: Codec

class RecordBatch:
    """A batch of rows that polars can import without copying."""

    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]: ...

class AvroIter:
    """An iterator over the record batches of an avro source."""

    def __iter__(self) -> AvroIter: ...
    def __next__(self) -> RecordBatch: ...

class AvroSource:
    """A pseudo-iterator over Avro files.

    ``paths`` are local files read natively. ``sources`` are factories, each
    returning a fresh single-use context manager whose ``__enter__`` yields a
    seekable binary file and whose ``__exit__`` releases it; the reader calls a
    factory once per scan (and re-call to rewind).
    """

    def __init__(
        self,
        sources: list[str | Callable[[], AbstractContextManager[BinaryIO]]],
        strict: bool,
        utf8_view: bool,
    ) -> None: ...
    def schema(self) -> RecordBatch: ...
    def batch_iter(
        self,
        batch_size: int,
        with_columns: list[str] | None,
    ) -> AvroIter: ...

class AvroFileSink:
    """A sink that writes tables to a file."""

    def __init__(
        self, path: str, schema: ArrowStreamExportable, codec: Codec | None
    ) -> None: ...
    def write(self, table: ArrowStreamExportable) -> None: ...
    def close(self) -> None: ...

class AvroBuffSink:
    """A sink that writes tables to a writable binary buffer."""

    def __init__(
        self, buff: BinaryIO, schema: ArrowStreamExportable, codec: Codec | None
    ) -> None: ...
    def write(self, table: ArrowStreamExportable) -> None: ...
    def close(self) -> None: ...

class AvroError(Exception):
    """An exception thrown from the native avro reader and writer."""

class EmptySources(ValueError):
    """An exception for when no sources are given."""

class AvroSpecError(ValueError):
    """An exception raised when data doesn't align to the avro spec."""
