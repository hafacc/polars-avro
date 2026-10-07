//! pyo3 bindings

use super::{Error, ReadOptions, Reader, Writer, get_schema};
use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow_avro::compression::CompressionCodec;
use pyo3::exceptions::{PyException, PyIOError, PyIndexError, PyKeyError, PyValueError};
use pyo3::types::{PyAnyMethods, PyBytes, PyBytesMethods, PyModule, PyModuleMethods};
use pyo3::{
    Bound, FromPyObject, Py, PyAny, PyErr, PyRef, PyResult, Python, create_exception, pyclass,
    pymethods, pymodule,
};
use pyo3_arrow::{PyRecordBatch, PyTable};
use std::convert::Infallible;
use std::error::Error as StdError;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::iter::Fuse;
use std::sync::Arc;

/// Hardcoded read buffer for python-backed (`PyIO`) sources.
///
/// Batches the per-read python callbacks (and cloud round-trips) into large
/// chunks. Local files don't need this — the OS already buffers them natively.
const PY_BUFFER_CAPACITY: usize = 4 * 1024 * 1024;

/// Recover a [`PyErr`] smuggled across the `std::io` boundary.
///
/// The exception rides as an [`io::Error`] payload, which arrow wraps exactly
/// once — so the arrow error's direct source is that [`io::Error`]. `None` if
/// there's no wrapped exception.
fn recover_py_err(err: &(dyn StdError + 'static)) -> Option<PyErr> {
    let io_err = err.source()?.downcast_ref::<io::Error>()?;
    let py_err = io_err.get_ref()?.downcast_ref::<PyErr>()?;
    Some(Python::attach(|py| py_err.clone_ref(py)))
}

/// A python file obtained by entering the context manager from a source factory.
///
/// The manager's `__exit__` is called when this is dropped, so cloud connections
/// are released promptly after each scan.
#[derive(Debug)]
struct EnteredSource {
    file: PyIO,
    ctx: Py<PyAny>,
}

impl Read for EnteredSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.file.read(buf)
    }
}

impl Seek for EnteredSource {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.file.seek(pos)
    }
}

impl Drop for EnteredSource {
    fn drop(&mut self) {
        Python::attach(|py| {
            let none = py.None();
            let _ = self
                .ctx
                .bind(py)
                .call_method1("__exit__", (&none, &none, &none));
        });
    }
}

#[derive(Debug)]
enum ScanSource {
    File(File),
    Opened(BufReader<EnteredSource>),
}

impl Read for ScanSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            ScanSource::File(reader) => reader.read(buf),
            ScanSource::Opened(reader) => reader.read(buf),
        }
    }
}

impl Seek for ScanSource {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match self {
            ScanSource::File(reader) => reader.seek(pos),
            ScanSource::Opened(reader) => reader.seek(pos),
        }
    }
}

/// A single scan source, extracted from python in caller order.
///
/// `Path` is tried first, so a python `str` is read as a local file (natively,
/// no GIL) and any other object is treated as a context-manager factory. Keeping
/// both kinds in one ordered list preserves the caller's source order, matching
/// polars.
#[derive(FromPyObject, Debug)]
enum Source {
    Path(String),
    Factory(Py<PyAny>),
}

/// Opens each [`Source`] into a [`ScanSource`] on demand, in order.
///
/// Local paths open natively; factories are called to get a fresh context
/// manager which is entered (`__enter__`) here and exited when the resulting
/// source is dropped.
#[derive(Debug)]
struct SourceIter {
    sources: Arc<[Source]>,
    idx: usize,
}

impl Iterator for SourceIter {
    type Item = Result<ScanSource, PyErr>;

    fn next(&mut self) -> Option<Self::Item> {
        let source = self.sources.get(self.idx)?;
        self.idx += 1;
        Some(match source {
            Source::Path(path) => match File::open(path) {
                Ok(file) => Ok(ScanSource::File(file)),
                Err(err) => Err(PyIOError::new_err(format!("I/O error: {path}: {err}"))),
            },
            Source::Factory(factory) => Python::attach(|py| {
                let ctx = factory.bind(py).call0()?;
                let file = ctx.call_method0("__enter__")?.unbind();
                let entered = EnteredSource {
                    file: PyIO(Arc::new(file)),
                    ctx: ctx.unbind(),
                };
                Ok(ScanSource::Opened(BufReader::with_capacity(
                    PY_BUFFER_CAPACITY,
                    entered,
                )))
            }),
        })
    }
}

#[pyclass]
#[derive(Debug)]
pub struct PyAvroIter(Fuse<Reader<ScanSource, SourceIter, Vec<String>>>);

#[pymethods]
impl PyAvroIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    // Returning `Ok(None)` raises `StopIteration`.
    fn __next__(&mut self) -> PyResult<Option<PyRecordBatch>> {
        let batch = self.0.next().transpose()?;
        Ok(batch.map(PyRecordBatch::from))
    }
}

#[derive(Debug, Clone)]
struct PyIO(Arc<Py<PyAny>>);

impl PyIO {
    // readonly seek
    fn py_seek(&self, pos: SeekFrom) -> io::Result<u64> {
        match pos {
            SeekFrom::Start(pos) => Python::attach(|py| {
                let writer = self.0.bind(py);
                let res = writer.call_method1("seek", (pos,))?;
                res.extract()
            })
            .map_err(io::Error::other::<PyErr>),
            SeekFrom::Current(offset) => Python::attach(|py| {
                let writer = self.0.bind(py);
                let res = writer.call_method0("tell")?;
                let current: u64 = res.extract()?;
                let pos = if offset < 0 {
                    current.saturating_sub(offset.unsigned_abs())
                } else {
                    current.saturating_add(offset.unsigned_abs())
                };
                let res = writer.call_method1("seek", (pos,))?;
                res.extract()
            })
            .map_err(io::Error::other::<PyErr>),
            SeekFrom::End(_) => Err(io::Error::new(
                ErrorKind::Unsupported,
                "seeking from end is not supported",
            )),
        }
    }
}

impl Read for PyIO {
    fn read(&mut self, mut buf: &mut [u8]) -> io::Result<usize> {
        Python::attach(|py| {
            let res = self.0.bind(py).call_method1("read", (buf.len(),))?;
            let bytes = res.cast_into::<PyBytes>()?;
            let raw = bytes.as_bytes();
            buf.write_all(raw)?;
            Ok(raw.len())
        })
        .map_err(io::Error::other::<PyErr>)
    }
}

impl Write for PyIO {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Python::attach(|py| {
            let res = self.0.bind(py).call_method1("write", (buf,))?;
            res.extract()
        })
        .map_err(io::Error::other::<PyErr>)
    }

    fn flush(&mut self) -> io::Result<()> {
        Python::attach(|py| {
            self.0.bind(py).call_method0("flush")?;
            Ok(())
        })
        .map_err(io::Error::other::<PyErr>)
    }
}

impl Seek for PyIO {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.py_seek(pos)
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct AvroSource {
    sources: Arc<[Source]>,
    schema: Option<SchemaRef>,
}

impl AvroSource {
    fn get_sources(&self) -> SourceIter {
        SourceIter {
            sources: self.sources.clone(),
            idx: 0,
        }
    }

    fn get_schema(&mut self) -> Result<SchemaRef, Error<PyErr>> {
        if let Some(schema) = &self.schema {
            Ok(schema.clone())
        } else {
            let first = self
                .get_sources()
                .next()
                .ok_or(Error::EmptySources)?
                .map_err(Error::User)?;
            let schema = get_schema(BufReader::new(first)).map_err(Error::widen)?;
            self.schema = Some(schema.clone());
            Ok(schema)
        }
    }
}

#[pymethods]
impl AvroSource {
    #[new]
    #[pyo3(signature = (sources))]
    fn new(sources: Vec<Source>) -> Self {
        Self {
            sources: sources.into(),
            schema: None,
        }
    }

    /// Return the file schema as a record batch without rows.
    #[pyo3(signature = ())]
    fn schema(&mut self) -> PyResult<PyRecordBatch> {
        Ok(RecordBatch::new_empty(self.get_schema()?).into())
    }

    #[pyo3(signature = (strict, utf8_view, batch_size, with_columns))]
    #[allow(clippy::needless_pass_by_value)]
    fn batch_iter(
        &mut self,
        strict: bool,
        utf8_view: bool,
        batch_size: usize,
        with_columns: Option<Vec<String>>,
    ) -> PyResult<PyAvroIter> {
        Ok(PyAvroIter(
            Reader::try_new(
                self.get_sources(),
                ReadOptions {
                    strict,
                    utf8_view,
                    batch_size,
                    projection: with_columns,
                },
            )?
            .fuse(),
        ))
    }
}

#[pyclass(eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    Deflate,
    Snappy,
    Bzip2,
    Xz,
    Zstandard,
}

impl From<Codec> for CompressionCodec {
    fn from(obj: Codec) -> Self {
        match obj {
            Codec::Deflate => CompressionCodec::Deflate,
            Codec::Snappy => CompressionCodec::Snappy,
            Codec::Bzip2 => CompressionCodec::Bzip2,
            Codec::Xz => CompressionCodec::Xz,
            Codec::Zstandard => CompressionCodec::ZStandard,
        }
    }
}

/// A sink writing avro to a local file.
#[pyclass]
pub struct AvroFileSink(Writer<BufWriter<File>>);

#[pymethods]
impl AvroFileSink {
    #[new]
    #[pyo3(signature = (path, schema, codec=None))]
    fn new(path: &str, schema: PyTable, codec: Option<Codec>) -> Result<Self, PyErr> {
        Ok(Self(Writer::try_new(
            BufWriter::new(File::create(path)?),
            schema.into_inner().1,
            codec.map(CompressionCodec::from),
        )?))
    }

    #[pyo3(signature = (table))]
    #[allow(clippy::needless_pass_by_value)]
    fn write(&mut self, table: PyTable) -> Result<(), PyErr> {
        for batch in table.batches() {
            self.0.write(batch)?;
        }
        Ok(())
    }

    #[pyo3(signature = ())]
    fn close(&mut self) -> Result<(), PyErr> {
        Ok(self.0.finish()?)
    }
}

/// A sink writing avro to a python binary buffer.
#[pyclass]
pub struct AvroBuffSink(Writer<BufWriter<PyIO>>);

#[pymethods]
impl AvroBuffSink {
    #[new]
    #[pyo3(signature = (buff, schema, codec=None))]
    fn new(buff: Py<PyAny>, schema: PyTable, codec: Option<Codec>) -> Result<Self, PyErr> {
        Ok(Self(Writer::try_new(
            BufWriter::new(PyIO(Arc::new(buff))),
            schema.into_inner().1,
            codec.map(CompressionCodec::from),
        )?))
    }

    #[pyo3(signature = (table))]
    #[allow(clippy::needless_pass_by_value)]
    fn write(&mut self, table: PyTable) -> Result<(), PyErr> {
        for batch in table.batches() {
            self.0.write(batch)?;
        }
        Ok(())
    }

    #[pyo3(signature = ())]
    fn close(&mut self) -> Result<(), PyErr> {
        Ok(self.0.finish()?)
    }
}

impl From<Error<PyErr>> for PyErr {
    fn from(value: Error<PyErr>) -> Self {
        match value {
            Error::User(err) => err,
            Error::Arrow(err) => {
                recover_py_err(&err).unwrap_or_else(|| AvroError::new_err(err.to_string()))
            }
            Error::ArrowAvro(err) => {
                recover_py_err(&err).unwrap_or_else(|| AvroError::new_err(err.to_string()))
            }
            Error::IO(err, path) => match err.downcast::<PyErr>() {
                Ok(py_err) => py_err,
                Err(err) => PyIOError::new_err(format!("I/O error: {path}: {err}")),
            },
            Error::Avro(err) => AvroError::new_err(err.to_string()),
            Error::Json(err) => AvroError::new_err(err.to_string()),
            Error::EmptySources => EmptySources::new_err("must scan at least one source"),
            Error::NonRecordSchema => {
                AvroSpecError::new_err("top level avro schema must be a record")
            }
            Error::LargeHeader => {
                AvroSpecError::new_err("header was too large to effectively parse")
            }
            err @ Error::NonMatchingSchemas { .. } => AvroSpecError::new_err(format!("{err}")),
            Error::ColumnNotFound(col) => {
                PyKeyError::new_err(format!("Column \"{col}\" not found in schema"))
            }
            Error::ColumnIndexOutOfBounds(ind) => {
                PyIndexError::new_err(format!("Column index {ind} is out of bounds"))
            }
        }
    }
}

impl From<Error<Infallible>> for PyErr {
    fn from(value: Error<Infallible>) -> Self {
        value.widen::<PyErr>().into()
    }
}

create_exception!(exceptions, AvroError, PyException);
create_exception!(exceptions, EmptySources, PyValueError);
create_exception!(exceptions, AvroSpecError, PyValueError);

#[pymodule]
fn _avro_rs(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<AvroSource>()?;
    m.add_class::<AvroFileSink>()?;
    m.add_class::<AvroBuffSink>()?;
    m.add_class::<Codec>()?;
    m.add("AvroError", py.get_type::<AvroError>())?;
    m.add("EmptySources", py.get_type::<EmptySources>())?;
    m.add("AvroSpecError", py.get_type::<AvroSpecError>())?;
    Ok(())
}
