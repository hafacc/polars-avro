#![feature(test)]

extern crate test;

use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow::datatypes::{Field, Schema};
use polars_avro::{Projection, ReadOptions, Reader, Writer};
use std::convert::Infallible;
use std::io::Cursor;
use std::sync::Arc;
use test::Bencher;

fn create_narrow_frame(n: i32) -> RecordBatch {
    RecordBatch::try_from_iter([
        (
            "idx",
            Arc::new(Int32Array::from_iter_values(0..n)) as ArrayRef,
        ),
        (
            "name",
            Arc::new(StringArray::from_iter_values((0..n).map(|v| v.to_string()))) as ArrayRef,
        ),
    ])
    .unwrap()
}

fn create_wide_frame(n: i32) -> RecordBatch {
    RecordBatch::try_from_iter([
        (
            "i32_col",
            Arc::new(Int32Array::from_iter_values(0..n)) as ArrayRef,
        ),
        (
            "i64_col",
            Arc::new(Int64Array::from_iter_values(
                (0..n).map(|v| i64::from(v) * 1000),
            )) as ArrayRef,
        ),
        (
            "f64_col",
            Arc::new(Float64Array::from_iter_values(
                (0..n).map(|v| f64::from(v) * 0.01),
            )) as ArrayRef,
        ),
        (
            "f32_col",
            Arc::new(Float32Array::from_iter_values(
                (0..n).map(|v| v as f32 * 0.1),
            )) as ArrayRef,
        ),
        (
            "bool_col",
            Arc::new(BooleanArray::from_iter((0..n).map(|v| Some(v % 2 == 0)))) as ArrayRef,
        ),
        (
            "str_col",
            Arc::new(StringArray::from_iter_values(
                (0..n).map(|v| format!("val_{v}")),
            )) as ArrayRef,
        ),
        (
            "binary_col",
            Arc::new(BinaryArray::from_iter_values(
                (0..n).map(|v| format!("bin_{v}").into_bytes()),
            )) as ArrayRef,
        ),
        (
            "str2_col",
            Arc::new(StringArray::from_iter_values(
                (0..n).map(|v| format!("extra_{v}")),
            )) as ArrayRef,
        ),
    ])
    .unwrap()
}

fn create_mega_wide_frame(n: i32) -> RecordBatch {
    let base = create_wide_frame(n);
    let mut fields = Vec::with_capacity(base.num_columns() * 16);
    let mut arrays = Vec::with_capacity(base.num_columns() * 16);
    for group in 0..16 {
        for (field, array) in base.schema().fields().iter().zip(base.columns()) {
            fields.push(Field::new(
                format!("{}_{group}", field.name()),
                field.data_type().clone(),
                field.is_nullable(),
            ));
            arrays.push(array.clone());
        }
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

macro_rules! bench_shape {
    ($name:ident, $frame_expr:expr) => {
        mod $name {
            use super::*;

            #[bench]
            fn write_polars_avro(b: &mut Bencher) {
                let frame = $frame_expr;
                b.iter(|| {
                    test::black_box(
                        Writer::try_new(Vec::new(), frame.schema(), None)
                            .unwrap()
                            .write(&frame)
                            .unwrap(),
                    )
                });
            }

            #[bench]
            fn read_polars_avro(b: &mut Bencher) {
                let frame = $frame_expr;
                let mut buff = Cursor::new(Vec::new());
                Writer::try_new(&mut buff, frame.schema(), None)
                    .unwrap()
                    .write(&frame)
                    .unwrap();
                b.iter(move || {
                    buff.set_position(0);
                    test::black_box(
                        Reader::try_new([Ok::<_, Infallible>(&mut buff)], ReadOptions::default())
                            .unwrap()
                            .collect::<Vec<_>>(),
                    )
                });
            }
        }
    };
}

macro_rules! bench_projection {
    ($name:ident, $frame_expr:expr, [$($col:literal),+ $(,)?]) => {
        mod $name {
            use super::*;

            #[bench]
            fn read_polars_avro(b: &mut Bencher) {
                let frame = $frame_expr;
                let mut buff = Cursor::new(Vec::new());
                Writer::try_new(&mut buff, frame.schema(), None)
                    .unwrap()
                    .write(&frame)
                    .unwrap();
                let columns: Vec<String> = vec![$($col.to_owned()),+];
                b.iter(move || {
                    buff.set_position(0);
                    test::black_box(
                        Reader::try_new(
                            [Ok::<_, Infallible>(&mut buff)],
                            ReadOptions {
                                projection: Some(Projection::Names(columns.clone())),
                                ..ReadOptions::default()
                            },
                        )
                        .unwrap()
                        .collect::<Vec<_>>(),
                    )
                });
            }
        }
    };
}

macro_rules! bench_read_options {
    ($name:ident, $frame_expr:expr, $($field:ident : $value:expr),+ $(,)?) => {
        mod $name {
            use super::*;

            #[bench]
            fn read_polars_avro(b: &mut Bencher) {
                let frame = $frame_expr;
                let mut buff = Cursor::new(Vec::new());
                Writer::try_new(&mut buff, frame.schema(), None)
                    .unwrap()
                    .write(&frame)
                    .unwrap();
                b.iter(move || {
                    buff.set_position(0);
                    test::black_box(
                        Reader::try_new(
                            [Ok::<_, Infallible>(&mut buff)],
                            ReadOptions {
                                $($field: $value,)*
                                ..ReadOptions::default()
                            },
                        )
                        .unwrap()
                        .collect::<Vec<_>>(),
                    )
                });
            }
        }
    };
}

bench_shape!(narrow_small, create_narrow_frame(16));
bench_shape!(narrow_medium, create_narrow_frame(1024));
bench_shape!(narrow_long, create_narrow_frame(65536));
bench_shape!(wide, create_wide_frame(1024));
bench_shape!(large, create_wide_frame(1_048_576));
bench_shape!(mega_wide, create_mega_wide_frame(1_048_576));

bench_read_options!(narrow_medium_strict, create_narrow_frame(1024), strict: true);
bench_read_options!(narrow_medium_utf8_view, create_narrow_frame(1024), utf8_view: true);

bench_projection!(
    wide_project_two,
    create_wide_frame(1024),
    ["i32_col", "str_col"]
);
bench_projection!(
    wide_project_five,
    create_wide_frame(1024),
    ["i32_col", "f64_col", "bool_col", "str_col", "binary_col"]
);
bench_projection!(
    mega_wide_project_eight,
    create_mega_wide_frame(1_048_576),
    [
        "i32_col_0",
        "i64_col_0",
        "f64_col_0",
        "f32_col_0",
        "bool_col_0",
        "str_col_0",
        "binary_col_0",
        "str2_col_0"
    ]
);
