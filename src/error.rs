//! Internal error types

use apache_avro::Error as AvroError;
use arrow::datatypes::Schema;
use arrow::error::ArrowError;
use arrow_avro::errors::AvroError as ArrowAvroError;
use std::collections::HashMap;
use std::convert::Infallible;
use std::error::Error as StdError;
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::io;
use std::sync::Arc;

/// Any error raised by this crate.
///
/// `E` is the caller-supplied source iterator's error, forwarded unchanged
/// through [`Error::User`]; it defaults to [`Infallible`].
#[non_exhaustive]
#[derive(Debug)]
pub enum Error<E = Infallible> {
    /// An error from the arrow library
    Arrow(ArrowError),
    /// An error from the arrow-avro library
    ArrowAvro(ArrowAvroError),
    /// An error from parsing the avro header
    Avro(AvroError),
    /// An error serializing a projected schema back to json
    Json(serde_json::Error),
    /// Cannot scan empty sources
    EmptySources,
    /// Top level avro schema must be a record
    NonRecordSchema,
    /// Happens when an avro header doesn't fit in an i64
    LargeHeader,
    /// If not all schemas in a batch were identical
    NonMatchingSchemas {
        /// The schema we expected (from the first source)
        expected: Schema,
        /// The schema we actually got
        actual: Arc<Schema>,
    },
    /// If a column wasn't found in the schema
    ColumnNotFound(String),
    /// Column index is out of bounds
    ColumnIndexOutOfBounds(usize),
    /// I/O related errors
    IO(io::Error, String),
    /// An error from the caller-supplied source iterator, forwarded unchanged
    User(E),
}

impl Error<Infallible> {
    /// Lift a source-free error into any `Error<E>`.
    ///
    /// Used where crate-internal code (which never produces a [`Error::User`])
    /// feeds a [`Reader`](crate::Reader) that carries a caller error type.
    #[must_use]
    pub fn widen<E>(self) -> Error<E> {
        match self {
            Error::Arrow(err) => Error::Arrow(err),
            Error::ArrowAvro(err) => Error::ArrowAvro(err),
            Error::Avro(err) => Error::Avro(err),
            Error::Json(err) => Error::Json(err),
            Error::EmptySources => Error::EmptySources,
            Error::NonRecordSchema => Error::NonRecordSchema,
            Error::LargeHeader => Error::LargeHeader,
            Error::NonMatchingSchemas { expected, actual } => {
                Error::NonMatchingSchemas { expected, actual }
            }
            Error::ColumnNotFound(col) => Error::ColumnNotFound(col),
            Error::ColumnIndexOutOfBounds(ind) => Error::ColumnIndexOutOfBounds(ind),
            Error::IO(err, path) => Error::IO(err, path),
            Error::User(never) => match never {},
        }
    }
}

impl<E: Display> Display for Error<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Error::Arrow(e) => write!(f, "Error from arrow: {e}"),
            Error::ArrowAvro(e) => write!(f, "Error from arrow-avro: {e}"),
            Error::Avro(e) => write!(f, "Error from avro: {e}"),
            Error::Json(e) => write!(f, "Error serializing schema: {e}"),
            Error::EmptySources => write!(f, "Cannot scan empty sources"),
            Error::NonRecordSchema => write!(f, "Top level avro schema must be a record"),
            Error::LargeHeader => write!(f, "Avro header is too large"),
            Error::NonMatchingSchemas { expected, actual } => {
                write!(f, "schemas differ:")?;
                let mut act_by_name: HashMap<_, _> = actual
                    .fields()
                    .iter()
                    .map(|f| (f.name().as_str(), f.data_type()))
                    .collect();
                for field in expected.fields() {
                    match act_by_name.remove(field.name().as_str()) {
                        None => write!(
                            f,
                            " removed \"{}\" ({:?}).",
                            field.name(),
                            field.data_type()
                        )?,
                        Some(act_dt) if act_dt != field.data_type() => write!(
                            f,
                            " \"{}\": expected {:?}, got {:?}.",
                            field.name(),
                            field.data_type(),
                            act_dt
                        )?,
                        _ => {}
                    }
                }
                for (name, dt) in &act_by_name {
                    write!(f, " added \"{name}\" ({dt:?}).")?;
                }
                Ok(())
            }
            Error::ColumnNotFound(col) => write!(f, "Column \"{col}\" wasn't found in the schema"),
            Error::ColumnIndexOutOfBounds(ind) => {
                write!(f, "Column index {ind} is out of bounds")
            }
            Error::IO(err, path) => write!(f, "Problem with {path}: {err}"),
            Error::User(err) => write!(f, "{err}"),
        }
    }
}

impl<E: StdError> StdError for Error<E> {}

impl<E> From<ArrowError> for Error<E> {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}

impl<E> From<ArrowAvroError> for Error<E> {
    fn from(value: ArrowAvroError) -> Self {
        Self::ArrowAvro(value)
    }
}

impl<E> From<AvroError> for Error<E> {
    fn from(value: AvroError) -> Self {
        Self::Avro(value)
    }
}

impl<E> From<serde_json::Error> for Error<E> {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl<E> From<io::Error> for Error<E> {
    fn from(value: io::Error) -> Self {
        Self::IO(value, "io".into())
    }
}

impl<E> From<Infallible> for Error<E> {
    fn from(value: Infallible) -> Self {
        match value {}
    }
}

#[cfg(test)]
mod tests {
    use super::Error as FullError;
    use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
    use arrow::error::ArrowError;
    use arrow_avro::errors::AvroError as ArrowAvroError;
    use std::convert::Infallible;
    use std::io;
    use std::sync::Arc;

    type Error = FullError<Infallible>;

    #[test]
    fn test_display() {
        let expected = Schema::new(vec![
            Field::new("kept", ArrowDataType::Int32, false),
            Field::new("changed", ArrowDataType::Int32, false),
            Field::new("removed", ArrowDataType::Float64, false),
        ]);
        let actual = Arc::new(Schema::new(vec![
            Field::new("kept", ArrowDataType::Int32, false),
            Field::new("changed", ArrowDataType::Utf8, false),
            Field::new("added", ArrowDataType::Boolean, false),
        ]));
        let avro_err = apache_avro::Schema::parse_str("not a schema").unwrap_err();
        let json_err = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        for err in [
            Error::Arrow(ArrowError::NotYetImplemented("test".into())),
            Error::ArrowAvro(ArrowAvroError::General("test".into())),
            Error::Avro(avro_err),
            Error::Json(json_err),
            Error::EmptySources,
            Error::NonRecordSchema,
            Error::LargeHeader,
            Error::NonMatchingSchemas { expected, actual },
            Error::ColumnNotFound("missing".into()),
            Error::ColumnIndexOutOfBounds(7),
            Error::IO(io::Error::other("boom"), "path".into()),
        ] {
            assert_ne!(format!("{err}"), "");
        }
    }

    #[test]
    fn test_from_conversions() {
        assert!(matches!(
            Error::from(ArrowError::NotYetImplemented("test".into())),
            Error::Arrow(_)
        ));
        assert!(matches!(
            Error::from(ArrowAvroError::General("test".into())),
            Error::ArrowAvro(_)
        ));
        assert!(matches!(
            Error::from(apache_avro::Schema::parse_str("not a schema").unwrap_err()),
            Error::Avro(_)
        ));
        assert!(matches!(
            Error::from(io::Error::other("boom")),
            Error::IO(_, _)
        ));
        assert!(matches!(
            Error::from(serde_json::from_str::<serde_json::Value>("{").unwrap_err()),
            Error::Json(_)
        ));
    }

    #[test]
    fn test_non_matching_display() {
        let expected = Schema::new(vec![
            Field::new("kept", ArrowDataType::Int32, false),
            Field::new("changed", ArrowDataType::Int32, false),
            Field::new("removed", ArrowDataType::Float64, false),
        ]);
        let actual = Arc::new(Schema::new(vec![
            Field::new("kept", ArrowDataType::Int32, false),
            Field::new("changed", ArrowDataType::Utf8, false),
            Field::new("added", ArrowDataType::Boolean, false),
        ]));
        let msg: String = format!("{}", Error::NonMatchingSchemas { expected, actual });
        assert!(msg.contains("removed \"removed\""), "{msg}");
        assert!(msg.contains("added \"added\""), "{msg}");
        assert!(msg.contains("\"changed\": expected"), "{msg}");
        assert!(!msg.contains("\"kept\""), "{msg}");
    }

    /// A `fmt::Write` that fails as soon as it sees a marker substring, used to
    /// drive the writer-error paths inside `Display`.
    struct FailOn(&'static str);

    impl std::fmt::Write for FailOn {
        fn write_str(&mut self, segment: &str) -> std::fmt::Result {
            if segment.contains(self.0) {
                Err(std::fmt::Error)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn test_non_matching_write_errors() {
        use std::fmt::Write as _;

        // failing while writing a removed field propagates the error
        let expected = Schema::new(vec![Field::new("gone", ArrowDataType::Int32, false)]);
        let actual = Arc::new(Schema::new(Vec::<Field>::new()));
        let removed = Error::NonMatchingSchemas { expected, actual };
        assert!(write!(FailOn("removed"), "{removed}").is_err());

        // failing while writing a changed field propagates the error
        let expected = Schema::new(vec![Field::new("col", ArrowDataType::Int32, false)]);
        let actual = Arc::new(Schema::new(vec![Field::new(
            "col",
            ArrowDataType::Utf8,
            false,
        )]));
        let changed = Error::NonMatchingSchemas { expected, actual };
        assert!(write!(FailOn("expected"), "{changed}").is_err());

        // failing on the leading "schemas differ:" write propagates the error
        let expected = Schema::new(vec![Field::new("col", ArrowDataType::Int32, false)]);
        let actual = Arc::new(Schema::new(vec![Field::new(
            "col",
            ArrowDataType::Int32,
            false,
        )]));
        let same = Error::NonMatchingSchemas { expected, actual };
        assert!(write!(FailOn("schemas"), "{same}").is_err());

        // failing while writing an added field propagates the error
        let expected = Schema::new(vec![Field::new("col", ArrowDataType::Int32, false)]);
        let actual = Arc::new(Schema::new(vec![
            Field::new("col", ArrowDataType::Int32, false),
            Field::new("extra", ArrowDataType::Int32, false),
        ]));
        let added = Error::NonMatchingSchemas { expected, actual };
        assert!(write!(FailOn("added"), "{added}").is_err());
    }
}
