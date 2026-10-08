from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from functools import partial
from types import TracebackType
from typing import BinaryIO
from urllib.parse import urlparse

import fsspec  # type: ignore[reportMissingTypeStubs]

# Returns a context manager yielding a seekable binary file; called once per scan.
SourceFactory = Callable[[], AbstractContextManager[BinaryIO]]


def is_url(path: str) -> bool:
    """Return whether ``path`` is a URL to open with fsspec rather than locally.

    Rust's native file opener only handles local paths, so anything with a real
    (multi-character) URL scheme is routed to fsspec. A single-character scheme
    is treated as a Windows drive letter, i.e. a local path.
    """
    return len(urlparse(path).scheme) > 1


class _SeekableSource(AbstractContextManager[BinaryIO]):
    """Re-enterable manager over a caller-provided seekable buffer.

    Captures the buffer's position at creation and rewinds to it on each enter,
    so a ``LazyFrame`` can be scanned repeatedly. Exiting is a no-op — the buffer
    belongs to the caller, so we must not close it.
    """

    def __init__(self, buf: BinaryIO) -> None:
        self._buf = buf
        self._pos = buf.tell()

    def __enter__(self) -> BinaryIO:
        self._buf.seek(self._pos)
        return self._buf

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        pass


def seekable_factory(buf: BinaryIO) -> SourceFactory:
    """Build a factory for a caller-provided seekable buffer.

    The position is captured once (now), not per call: re-creating the manager
    each scan would `tell` a buffer left at EOF by the previous scan. So we make
    one re-enterable manager and hand it back each time.
    """
    source = _SeekableSource(buf)
    return lambda: source


def cloud_factory(url: str, storage_options: Mapping[str, str]) -> SourceFactory:
    """Build a factory that opens ``url`` with fsspec, fresh per scan.

    ``fsspec.open`` returns an ``OpenFile``, which is itself the context manager
    (enter opens the file, exit closes it), so the factory is just a bound call.
    """
    return partial(fsspec.open, url, **storage_options)  # type: ignore[reportUnknownMemberType, reportReturnType]
