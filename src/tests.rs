use super::{ReadOptions, Reader, Writer, get_schema};
use arrow::array::{
    Array, BinaryArray, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow::compute::concat_batches;
use std::convert::Infallible;
use std::io::Cursor;
use std::sync::Arc;

#[allow(clippy::unnecessary_wraps)]
fn ok<T>(val: T) -> Result<T, Infallible> {
    Ok(val)
}

fn serialize(batch: &RecordBatch) -> Vec<u8> {
    let mut buff = Cursor::new(Vec::new());
    let mut writer = Writer::try_new(&mut buff, batch.schema(), None).unwrap();
    writer.write(batch).unwrap();
    writer.into_inner().unwrap();
    buff.into_inner()
}

fn deserialize(buff: Vec<u8>) -> RecordBatch {
    let mut cursor = Cursor::new(buff);
    let schema = get_schema(&mut cursor).unwrap();
    cursor.set_position(0);
    let reader = Reader::try_new(
        [ok(cursor)],
        ReadOptions {
            batch_size: 2,
            ..ReadOptions::default()
        },
    )
    .unwrap();
    let batches: Vec<RecordBatch> = reader.map(|batch| batch.unwrap()).collect();
    if batches.is_empty() {
        RecordBatch::new_empty(schema)
    } else {
        concat_batches(&batches[0].schema(), &batches).unwrap()
    }
}

/// Compare two batches by column values and names, ignoring field-level
/// metadata and nullability flags that avro doesn't round-trip.
fn assert_batches_equal(left: &RecordBatch, right: &RecordBatch) {
    assert_eq!(left.num_rows(), right.num_rows(), "row count mismatch");
    assert_eq!(
        left.num_columns(),
        right.num_columns(),
        "column count mismatch"
    );
    for (lhs, rhs) in left.schema().fields().iter().zip(right.schema().fields()) {
        assert_eq!(lhs.name(), rhs.name(), "column name mismatch");
    }
    for index in 0..left.num_columns() {
        assert_eq!(
            left.column(index).to_data(),
            right.column(index).to_data(),
            "column {:?} differs",
            left.schema().field(index).name()
        );
    }
}

/// Round trip a batch of the common flat avro types, including nulls.
#[test]
fn test_roundtrip_flat() {
    let batch = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int32Array::from(vec![Some(1), None, Some(3), Some(4)])) as Arc<dyn Array>,
        ),
        (
            "big",
            Arc::new(Int64Array::from(vec![
                Some(10_i64),
                Some(-20),
                None,
                Some(42),
            ])) as Arc<dyn Array>,
        ),
        (
            "small",
            Arc::new(Float32Array::from(vec![
                Some(1.5_f32),
                None,
                Some(3.5),
                Some(4.5),
            ])) as Arc<dyn Array>,
        ),
        (
            "score",
            Arc::new(Float64Array::from(vec![
                None,
                Some(72.5_f64),
                Some(53.6),
                None,
            ])) as Arc<dyn Array>,
        ),
        (
            "name",
            Arc::new(StringArray::from(vec![
                Some("alice"),
                Some("ben"),
                None,
                Some("dave"),
            ])) as Arc<dyn Array>,
        ),
        (
            "good",
            Arc::new(BooleanArray::from(vec![
                Some(true),
                Some(false),
                None,
                Some(true),
            ])) as Arc<dyn Array>,
        ),
        (
            "code",
            Arc::new(BinaryArray::from_opt_vec(vec![
                Some(&b"al1c3"[..]),
                Some(&b"b3n"[..]),
                Some(&b"chl03"[..]),
                None,
            ])) as Arc<dyn Array>,
        ),
    ])
    .unwrap();

    let reconstruction = deserialize(serialize(&batch));
    assert_batches_equal(&batch, &reconstruction);
}

/// Writing and reading back a zero-row batch preserves the schema.
#[test]
fn test_roundtrip_empty() {
    let batch = RecordBatch::try_from_iter([(
        "weight",
        Arc::new(Float64Array::from(Vec::<f64>::new())) as Arc<dyn Array>,
    )])
    .unwrap();
    let reconstruction = deserialize(serialize(&batch));
    assert_eq!(reconstruction.num_rows(), 0);
    assert_eq!(reconstruction.schema().field(0).name(), "weight");
}
