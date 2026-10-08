from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from os import path
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Self

from polars import DataFrame, Schema
from polars._typing import ArrowStreamExportable

from ._avro_rs import AvroBuffSink, AvroFileSink, Codec


def create_writer(
    schema: ArrowStreamExportable,
    *,
    dest: str | Path | BinaryIO,
    codec: Codec | None = None,
) -> AvroBuffSink | AvroFileSink:
    """Create a sink writing avro records matching ``schema``."""
    match dest:
        case str() | Path():
            expanded = path.expanduser(path.expandvars(dest))
            return AvroFileSink(expanded, schema, codec)
        case _:
            return AvroBuffSink(dest, schema, codec)


class AvroWriter:
    """Incrementally write DataFrames to an Avro file.

    Most polars types write directly (narrow ints widen, Time truncates to
    microseconds); only Categorical, Enum, and out-of-range UInt64 values must
    be cast before writing — see the README for workarounds.
    """

    def __init__(
        self,
        dest: str | Path | BinaryIO,
        *,
        schema: Schema | None = None,
        codec: Codec | None = None,
    ) -> None:
        self._create: Callable[[ArrowStreamExportable], AvroBuffSink | AvroFileSink] = (
            partial(
                create_writer,
                dest=dest,
                codec=codec,
            )
        )
        self._sink: AvroBuffSink | AvroFileSink | None = (
            None
            if schema is None
            else self._create(DataFrame(schema=schema).to_arrow())
        )

    def __enter__(self) -> Self:
        return self

    def write(self, batch: DataFrame) -> None:
        # arrow-rs rejects polars' own export of Null arrays (pola-rs/polars#22934)
        table = batch.to_arrow()
        if self._sink is None:
            self._sink = self._create(table)
        self._sink.write(table)

    def close(self) -> None:
        if self._sink is None:
            raise ValueError(
                "cannot write an avro file without any batches unless a schema "
                "is provided"
            )
        else:
            self._sink.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        elif self._sink is not None:
            # don't mask original exception
            self._sink.close()


def write_avro(
    batches: DataFrame | Iterable[DataFrame],
    dest: str | Path | BinaryIO,
    *,
    schema: Schema | None = None,
    codec: Codec | None = None,
) -> None:
    """Write a DataFrame or iterable of DataFrames to an Avro file.

    Most polars types write directly (narrow ints widen, Time truncates to
    microseconds); only Categorical, Enum, and out-of-range UInt64 values must
    be cast before writing — see the README for workarounds.

    Parameters
    ----------
    batches : A DataFrame or iterable of DataFrames to write.
    dest : The file path or writable binary buffer to write to.
    schema : The schema to use. If None, inferred from the first batch.
    codec : The compression codec to use, or None for no compression.
    """
    with AvroWriter(
        dest,
        schema=schema,
        codec=codec,
    ) as writer:
        if isinstance(batches, DataFrame):
            writer.write(batches)
        else:
            for batch in batches:
                writer.write(batch)
