//! Rust scan implementation

use super::Error;
use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
use arrow_avro::reader::{Reader as ArrowAvroReader, ReaderBuilder, read_header_info};
use std::io::{BufRead, Seek};
use std::iter::FusedIterator;
use std::sync::Arc;

/// The columns to read, in output order
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Projection {
    /// Columns by name
    Names(Vec<String>),
    /// Columns by position in the file
    Indices(Vec<usize>),
}

/// Configuration options for the avro reader
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadOptions {
    /// Enable stricter avro union handling: reject unions where `null` is not
    /// the first branch (`[T, "null"]` rather than `["null", T]`) instead of
    /// accepting them.
    pub strict: bool,
    /// If strings should be read in as views instead of character arrays.
    ///
    /// This affects UUID and nullable string handling — see the README for details.
    /// String views avoid copying, so enabling this is likely faster if you
    /// don't mind losing null string distinctions.
    pub utf8_view: bool,
    /// The batch size for reading
    pub batch_size: usize,
    /// The columns to select
    pub projection: Option<Projection>,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            strict: false,
            utf8_view: false,
            batch_size: 1024,
            projection: None,
        }
    }
}

impl ReadOptions {
    fn builder(&self) -> ReaderBuilder {
        ReaderBuilder::new()
            .with_utf8_view(self.utf8_view)
            .with_strict_mode(self.strict)
            .with_batch_size(self.batch_size)
    }

    /// Get the arrow schema these options read every column of an avro file as
    ///
    /// # Errors
    /// If the avro schema can't be read, or any errors from the reader
    pub fn schema<R: BufRead>(&self, reader: R) -> Result<SchemaRef, Error> {
        Ok(self.builder().build(reader)?.schema())
    }

    fn create_reader<R: BufRead + Seek>(&self, mut reader: R) -> Result<ArrowAvroReader<R>, Error> {
        let builder = match &self.projection {
            None => self.builder(),
            Some(Projection::Indices(indices)) => self.builder().with_projection(indices.clone()),
            Some(Projection::Names(names)) => {
                // arrow-avro selects by position, and only the header maps names to one
                let header = read_header_info(&mut reader)?;
                let rewind = i64::try_from(header.header_len()).map_err(|_| Error::LargeHeader)?;
                // a BufReader rewinds inside its buffer when the header fits in it
                reader.seek_relative(-rewind)?;
                let schema = self.builder().build(&mut reader)?.schema();
                reader.seek_relative(-rewind)?;
                let indices = names.iter().map(|name| {
                    schema
                        .index_of(name)
                        .map_err(|_| Error::ColumnNotFound(name.clone()))
                });
                self.builder()
                    .with_projection(indices.collect::<Result<_, _>>()?)
            }
        };
        Ok(builder.build(reader)?)
    }
}

/// An iterator that yields [`RecordBatch`]es from one or more avro sources.
///
/// All sources must share the same schema; a [`Error::NonMatchingSchemas`]
/// error is returned if they differ.
///
/// Sources are read as given, so wrap an unbuffered one (like a
/// [`File`](std::fs::File)) in a [`BufReader`](std::io::BufReader).
#[derive(Debug)]
pub struct Reader<R: BufRead, I> {
    sources: I,
    source: ArrowAvroReader<R>,
    options: ReadOptions,
    schema: Arc<Schema>,
}

impl<R, E, I> Reader<R, I>
where
    R: BufRead + Seek,
    I: Iterator<Item = Result<R, E>>,
{
    /// Create a new iterator from sources and a config
    ///
    /// # Errors
    /// If sources is empty, or is a problem creating a reader from the first
    /// source
    pub fn try_new(
        sources: impl IntoIterator<IntoIter = I>,
        config: ReadOptions,
    ) -> Result<Self, Error<E>> {
        let mut sources = sources.into_iter();
        let first = sources
            .next()
            .ok_or(Error::EmptySources)?
            .map_err(Error::User)?;
        let source = config.create_reader(first).map_err(Error::widen)?;
        let schema = source.schema();
        Ok(Self {
            sources,
            source,
            options: config,
            schema,
        })
    }

    fn matched_schema(&self, batch: &RecordBatch) -> bool {
        batch.schema() == self.schema
    }
}

impl<R, E, I> Iterator for Reader<R, I>
where
    R: BufRead + Seek,
    I: Iterator<Item = Result<R, E>>,
{
    type Item = Result<RecordBatch, Error<E>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.source.next() {
                Some(Ok(batch)) if self.matched_schema(&batch) => {
                    return Some(Ok(batch));
                }
                Some(Ok(batch)) => {
                    return Some(Err(Error::NonMatchingSchemas {
                        expected: (*self.schema).clone(),
                        actual: batch.schema(),
                    }));
                }
                Some(Err(e)) => return Some(Err(e.into())),
                None => match self.sources.next() {
                    Some(Ok(source)) => {
                        self.source = match self.options.create_reader(source) {
                            Ok(reader) => reader,
                            Err(e) => return Some(Err(e.widen())),
                        };
                    }
                    Some(Err(e)) => return Some(Err(Error::User(e))),
                    None => return None,
                },
            }
        }
    }
}

impl<R, E, I> FusedIterator for Reader<R, I>
where
    R: BufRead + Seek,
    I: Iterator<Item = Result<R, E>> + FusedIterator,
{
}

#[cfg(test)]
mod tests {
    use super::{Error, Projection, ReadOptions, Reader};
    use apache_avro::schema::{
        DecimalSchema, FixedSchema, InnerDecimalSchema, Name, RecordField, Schema, UnionSchema,
        UuidSchema,
    };
    use apache_avro::types::{Record, Value};
    use apache_avro::{
        AvroResult, Days, Decimal as AvroDecimal, Duration as AvroDuration, Millis, Months, Writer,
    };
    use arrow::array::{Array, RecordBatch};
    use arrow::compute::concat_batches;
    use arrow::datatypes::{DataType, IntervalUnit};
    use std::convert::Infallible;
    use std::error::Error as StdError;
    use std::fs::File;
    use std::io::{BufRead, BufReader, Cursor, Read, Seek};
    use std::mem;
    use uuid::Uuid;

    #[allow(clippy::unnecessary_wraps)]
    fn ok<T>(val: T) -> Result<T, Infallible> {
        Ok(val)
    }

    fn names(columns: &[&str]) -> Projection {
        let columns = columns.iter().map(|column| (*column).to_owned());
        Projection::Names(columns.collect())
    }

    /// Drain a reader and concatenate all of its batches into one.
    fn collect_one<R, E, I>(reader: Reader<R, I>) -> RecordBatch
    where
        R: BufRead + Seek,
        E: StdError,
        I: Iterator<Item = Result<R, E>>,
    {
        let batches: Vec<RecordBatch> = reader.map(|batch| batch.unwrap()).collect();
        let schema = batches
            .first()
            .expect("expected at least one batch")
            .schema();
        concat_batches(&schema, &batches).unwrap()
    }

    fn is_string_type(dtype: &DataType) -> bool {
        matches!(
            dtype,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        )
    }

    fn is_binary_type(dtype: &DataType) -> bool {
        matches!(
            dtype,
            DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_)
        )
    }

    /// Write a single-field avro record file to an in-memory buffer.
    fn write_avro(
        name: &str,
        dtype: Schema,
        vals: impl IntoIterator<Item = impl Into<Value>>,
    ) -> AvroResult<Cursor<Vec<u8>>> {
        let mut buff = Cursor::new(Vec::new());
        let schema = Schema::record(Name::new("base")?)
            .fields(vec![
                RecordField::builder().name(name).schema(dtype).build(),
            ])
            .build();
        let mut writer = Writer::new(&schema, &mut buff)?;
        for val in vals {
            let mut first = Record::new(&schema).unwrap();
            first.put(name, val);
            writer.append_value(first)?;
        }
        writer.flush()?;
        mem::drop(writer);
        buff.set_position(0);
        Ok(buff)
    }

    /// Test scan on a simple file
    #[test]
    fn test_scan() {
        let batches = Reader::try_new(
            [File::open("./resources/food.avro").map(BufReader::new)],
            ReadOptions::default(),
        )
        .unwrap();
        let frame = collect_one(batches);
        assert_eq!(frame.num_rows(), 27);
        assert_eq!(frame.num_columns(), 4);
    }

    /// Projection reorders and subsets the columns
    #[test]
    fn test_reorder() {
        let columns = ["sugars_g", "calories"];
        let batches = Reader::try_new(
            [File::open("./resources/food.avro").map(BufReader::new)],
            ReadOptions {
                projection: Some(names(&columns)),
                ..ReadOptions::default()
            },
        )
        .unwrap();
        let frame = collect_one(batches);
        let schema = frame.schema();
        let names: Vec<&str> = schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect();
        assert_eq!(names, columns);
    }

    /// Columns can be selected by position
    #[test]
    fn test_reorder_by_position() {
        let batches = Reader::try_new(
            [File::open("./resources/food.avro").map(BufReader::new)],
            ReadOptions {
                projection: Some(Projection::Indices(vec![3, 1])),
                ..ReadOptions::default()
            },
        )
        .unwrap();
        let schema = collect_one(batches).schema();
        let names: Vec<&str> = schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect();
        assert_eq!(names, ["sugars_g", "calories"]);
    }

    #[test]
    fn test_position_out_of_bounds_error() {
        let res = Reader::try_new(
            [File::open("./resources/food.avro").map(BufReader::new)],
            ReadOptions {
                projection: Some(Projection::Indices(vec![99])),
                ..ReadOptions::default()
            },
        );
        assert!(matches!(res, Err(Error::Arrow(_))));
    }

    /// Avro bytes are read as an arrow binary type
    #[test]
    fn test_bytes() {
        let buff = write_avro("bytes", Schema::Bytes, [&b"test"[..], &b"another"[..]]).unwrap();
        let frame = collect_one(Reader::try_new([ok(buff)], ReadOptions::default()).unwrap());
        assert_eq!(frame.num_rows(), 2);
        assert!(is_binary_type(frame.column(0).data_type()));
    }

    /// Avro fixed is read as a fixed size binary
    #[test]
    fn test_fixed() {
        let buff = write_avro(
            "fixed",
            Schema::fixed(Name::new("fixed").unwrap(), 4).build(),
            [
                Value::Fixed(4, vec![1, 2, 3, 4]),
                Value::Fixed(4, vec![5, 6, 7, 8]),
            ],
        )
        .unwrap();
        let frame = collect_one(Reader::try_new([ok(buff)], ReadOptions::default()).unwrap());
        assert_eq!(frame.column(0).data_type(), &DataType::FixedSizeBinary(4));
    }

    /// Avro decimal is read as a 128-bit decimal
    #[test]
    fn test_decimal() {
        let buff = write_avro(
            "decimal",
            Schema::Decimal(DecimalSchema {
                precision: 10,
                scale: 2,
                inner: InnerDecimalSchema::Bytes,
            }),
            [
                Value::Decimal(AvroDecimal::from(vec![0x64u8])),
                Value::Decimal(AvroDecimal::from(vec![0x00, 0xFA])),
            ],
        )
        .unwrap();
        let frame = collect_one(Reader::try_new([ok(buff)], ReadOptions::default()).unwrap());
        assert_eq!(frame.column(0).data_type(), &DataType::Decimal128(10, 2));
    }

    /// With `utf8_view` off, avro UUIDs decode to a binary type
    #[test]
    fn test_uuid_binary() {
        let buff = write_avro(
            "uuid",
            Schema::Uuid(UuidSchema::String),
            [Value::Uuid(
                Uuid::parse_str("936da01f-9abd-4d9d-80c7-02af85c822a8").unwrap(),
            )],
        )
        .unwrap();
        let frame = collect_one(
            Reader::try_new(
                [ok(buff)],
                ReadOptions {
                    utf8_view: false,
                    ..ReadOptions::default()
                },
            )
            .unwrap(),
        );
        assert!(is_binary_type(frame.column(0).data_type()));
    }

    /// With `utf8_view` on, avro UUIDs decode to a string type
    #[test]
    fn test_uuid_view() {
        let buff = write_avro(
            "uuid",
            Schema::Uuid(UuidSchema::String),
            [Value::Uuid(
                Uuid::parse_str("936da01f-9abd-4d9d-80c7-02af85c822a8").unwrap(),
            )],
        )
        .unwrap();
        let frame = collect_one(
            Reader::try_new(
                [ok(buff)],
                ReadOptions {
                    utf8_view: true,
                    ..ReadOptions::default()
                },
            )
            .unwrap(),
        );
        assert!(is_string_type(frame.column(0).data_type()));
    }

    /// With `utf8_view` off, nulls in a nullable string are preserved
    #[test]
    fn test_null_string_preserved() {
        let buff = write_avro(
            "val",
            Schema::Union(UnionSchema::new(vec![Schema::Null, Schema::String]).unwrap()),
            [Some("string"), None],
        )
        .unwrap();
        let frame = collect_one(
            Reader::try_new(
                [ok(buff)],
                ReadOptions {
                    utf8_view: false,
                    ..ReadOptions::default()
                },
            )
            .unwrap(),
        );
        assert!(is_string_type(frame.column(0).data_type()));
        assert_eq!(frame.column(0).null_count(), 1);
    }

    /// With `utf8_view` on, nulls in a nullable string become empty strings
    #[test]
    fn test_null_string_lossy() {
        let buff = write_avro(
            "val",
            Schema::Union(UnionSchema::new(vec![Schema::Null, Schema::String]).unwrap()),
            [Some("string"), None],
        )
        .unwrap();
        let frame = collect_one(
            Reader::try_new(
                [ok(buff)],
                ReadOptions {
                    utf8_view: true,
                    ..ReadOptions::default()
                },
            )
            .unwrap(),
        );
        assert!(is_string_type(frame.column(0).data_type()));
        assert_eq!(frame.column(0).null_count(), 0);
    }

    /// A non-first source that fails surfaces as an error item during iteration.
    #[test]
    fn test_source_error_propagates() {
        let valid = write_avro("col", Schema::Int, [1, 2, 3]).unwrap();
        let sources: Vec<Result<Cursor<Vec<u8>>, std::io::Error>> =
            vec![Ok(valid), Err(std::io::Error::other("boom"))];
        let mut reader = Reader::try_new(sources, ReadOptions::default()).unwrap();
        let last = reader.by_ref().last().unwrap();
        assert!(matches!(last, Err(Error::User(_))));
    }

    /// Serves a valid avro stream until `fail_at`, then returns a single I/O
    /// error followed by EOF, so the reader errors once and then stops.
    struct FailOnceReader {
        data: Cursor<Vec<u8>>,
        fail_at: u64,
        failed: bool,
    }

    impl Read for FailOnceReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let pos = self.data.position();
            if pos >= self.fail_at {
                if self.failed {
                    return Ok(0);
                }
                self.failed = true;
                return Err(std::io::Error::other("boom"));
            }
            let remaining = usize::try_from(self.fail_at - pos).unwrap_or(usize::MAX);
            let limit = remaining.min(buf.len());
            self.data.read(&mut buf[..limit])
        }
    }

    impl Seek for FailOnceReader {
        fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
            self.data.seek(pos)
        }
    }

    /// An I/O error raised by the underlying reader mid-stream is surfaced as an
    /// error item.
    ///
    /// The file is larger than the `BufReader` buffer so the header is read up
    /// front (letting construction succeed) while the failure lands on a later
    /// block read.
    #[test]
    fn test_reader_error_propagates() {
        let bytes = write_avro("col", Schema::Int, 0..20_000)
            .unwrap()
            .into_inner();
        assert!(bytes.len() > 8192, "need a multi-buffer file");
        let source = BufReader::new(FailOnceReader {
            fail_at: u64::try_from(bytes.len()).unwrap() / 2,
            data: Cursor::new(bytes),
            failed: false,
        });
        let mut reader = Reader::try_new(
            [ok(source)],
            ReadOptions {
                batch_size: 2,
                ..ReadOptions::default()
            },
        )
        .unwrap();
        // read up to (and including) the first error, then stop
        let err = reader
            .find(Result::is_err)
            .expect("expected an error item")
            .unwrap_err();
        assert!(matches!(err, Error::Arrow(_)), "{err:?}");
    }

    #[test]
    fn test_empty_sources_error() {
        let sources: Vec<Result<Cursor<Vec<u8>>, std::io::Error>> = Vec::new();
        let err = Reader::try_new(sources, ReadOptions::default()).unwrap_err();
        assert!(matches!(err, Error::EmptySources));
    }

    #[test]
    fn test_first_source_error() {
        let sources: Vec<Result<Cursor<Vec<u8>>, std::io::Error>> =
            vec![Err(std::io::Error::other("boom"))];
        let err = Reader::try_new(sources, ReadOptions::default()).unwrap_err();
        assert!(matches!(err, Error::User(_)));
    }

    #[test]
    fn test_projection_bad_header_error() {
        let err = Reader::try_new(
            [ok(Cursor::new(b"not an avro file".to_vec()))],
            ReadOptions {
                projection: Some(names(&["x"])),
                ..ReadOptions::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::Arrow(_) | Error::ArrowAvro(_)),
            "{err:?}"
        );
    }

    /// Serves valid data but fails every seek.
    #[derive(Debug)]
    struct NoSeek(Cursor<Vec<u8>>);

    impl Read for NoSeek {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.0.read(buf)
        }
    }

    impl Seek for NoSeek {
        fn seek(&mut self, _: std::io::SeekFrom) -> std::io::Result<u64> {
            Err(std::io::Error::other("no seek"))
        }
    }

    fn project_without_seeking(name: &str) -> Result<RecordBatch, Error> {
        let valid = write_avro(name, Schema::Int, [1, 2, 3])
            .unwrap()
            .into_inner();
        let reader = Reader::try_new(
            [ok(BufReader::new(NoSeek(Cursor::new(valid))))],
            ReadOptions {
                projection: Some(names(&[name])),
                ..ReadOptions::default()
            },
        )?;
        Ok(collect_one(reader))
    }

    /// Selecting columns rewinds inside the buffer, without seeking the source.
    #[test]
    fn test_projection_doesnt_seek() {
        let frame = project_without_seeking("col").unwrap();
        assert_eq!(frame.num_rows(), 3);
    }

    #[test]
    fn test_projection_large_header_rewind_error() {
        // a long field name makes the header exceed the 8 KiB BufReader buffer
        let err = project_without_seeking(&"f".repeat(9000)).unwrap_err();
        assert!(matches!(err, Error::IO(_, _)), "{err:?}");
    }

    /// Avro duration is read as a month/day/nano interval
    #[test]
    fn test_duration() {
        let buff = write_avro(
            "duration",
            Schema::Duration(
                FixedSchema::builder()
                    .name(Name::new("duration").unwrap())
                    .size(12)
                    .build(),
            ),
            [Value::Duration(AvroDuration::new(
                Months::new(1),
                Days::new(2),
                Millis::new(3),
            ))],
        )
        .unwrap();
        let frame = collect_one(Reader::try_new([ok(buff)], ReadOptions::default()).unwrap());
        assert_eq!(
            frame.column(0).data_type(),
            &DataType::Interval(IntervalUnit::MonthDayNano)
        );
    }

    /// Root Avro schema must be a Record
    #[test]
    fn test_single_column_error() {
        let mut buff = Cursor::new(Vec::new());
        let mut writer = Writer::new(&Schema::Int, &mut buff).unwrap();
        writer.append_value(1).unwrap();
        writer.append_value(2).unwrap();
        writer.flush().unwrap();
        mem::drop(writer);
        buff.set_position(0);
        let err = Reader::try_new([ok(buff)], ReadOptions::default()).unwrap_err();
        assert!(matches!(err, Error::Arrow(_)));
    }

    /// A projected column that isn't present errors out
    #[test]
    fn test_missing_columns_error() {
        let res = Reader::try_new(
            [File::open("./resources/food.avro").map(BufReader::new)],
            ReadOptions {
                projection: Some(names(&["missing"])),
                ..ReadOptions::default()
            },
        );
        assert!(matches!(res, Err(Error::ColumnNotFound(_))));
    }

    #[test]
    fn test_different_schemas() {
        let one = write_avro("x", Schema::Int, [1, 2, 3]).unwrap();
        let two = write_avro("y", Schema::String, ["a", "b", "c"]).unwrap();

        let iter = Reader::try_new(
            [ok(one), ok(two)],
            ReadOptions {
                batch_size: 2,
                ..ReadOptions::default()
            },
        )
        .unwrap();
        let err = iter.collect::<Result<Vec<_>, _>>().unwrap_err();
        assert!(matches!(err, Error::NonMatchingSchemas { .. }));
    }

    #[test]
    fn test_different_schemas_projection() {
        let one = write_avro("x", Schema::Int, [1, 2, 3]).unwrap();
        let two = write_avro("y", Schema::String, ["a", "b", "c"]).unwrap();

        let iter = Reader::try_new(
            [ok(one), ok(two)],
            ReadOptions {
                batch_size: 2,
                projection: Some(names(&["x"])),
                ..ReadOptions::default()
            },
        )
        .unwrap();
        let err = iter.collect::<Result<Vec<_>, _>>().unwrap_err();
        assert!(matches!(err, Error::ColumnNotFound(_)));
    }

    /// Sources whose selected column differs in type are caught under a projection.
    #[test]
    fn test_different_types_projection() {
        let one = write_avro("x", Schema::Int, [1, 2, 3]).unwrap();
        let two = write_avro("x", Schema::String, ["a", "b", "c"]).unwrap();

        let iter = Reader::try_new(
            [ok(one), ok(two)],
            ReadOptions {
                batch_size: 2,
                projection: Some(names(&["x"])),
                ..ReadOptions::default()
            },
        )
        .unwrap();
        let err = iter.collect::<Result<Vec<_>, _>>().unwrap_err();
        assert!(matches!(err, Error::NonMatchingSchemas { .. }), "{err:?}");
    }

    /// Projecting a logical-typed column keeps its logical arrow type instead of
    /// decoding it as the raw underlying primitive.
    #[test]
    fn test_projection_preserves_logical_type() {
        let buff = write_avro("d", Schema::Date, [Value::Date(18262)]).unwrap();
        let frame = collect_one(
            Reader::try_new(
                [ok(buff)],
                ReadOptions {
                    projection: Some(names(&["d"])),
                    ..ReadOptions::default()
                },
            )
            .unwrap(),
        );
        assert_eq!(frame.column(0).data_type(), &DataType::Date32);
    }
}
