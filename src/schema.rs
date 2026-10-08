use super::Error;
use arrow::datatypes::SchemaRef;
use arrow_avro::reader::ReaderBuilder;
use std::io::BufRead;

/// Get an arrow schema from an avro reader
///
/// # Errors
/// If the avro schema can't be read, or any errors from the reader
pub fn get_schema<R: BufRead>(reader: R) -> Result<SchemaRef, Error> {
    let reader = ReaderBuilder::new().build(reader)?;
    Ok(reader.schema())
}

#[cfg(test)]
mod tests {
    use super::super::Error;
    use super::get_schema;

    #[test]
    fn test_get_schema_bad_header() {
        let err = get_schema(&b"this is not an avro file"[..]).unwrap_err();
        assert!(
            matches!(err, Error::Arrow(_) | Error::ArrowAvro(_)),
            "{err:?}"
        );
    }
}
