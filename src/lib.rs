//! Reading and writing [Apache Avro](https://avro.apache.org/) files as Arrow
//! record batches.
//!
//! Not all arrow types can be written to avro. See [`Writer`] for types that
//! require casting before writing. When reading, some avro types map to
//! different arrow types than you might expect — see [`ReadOptions`] for
//! details on `utf8_view` and type mapping behavior.

#![warn(clippy::pedantic)]
#![warn(clippy::too_long_first_doc_paragraph)]
#![warn(missing_docs)]

mod error;
#[cfg(feature = "pyo3")]
mod py;
mod scan;
mod schema;
mod sink;
#[cfg(test)]
mod tests;

/// Avro compression codecs for writing.
pub use arrow_avro::compression::CompressionCodec;
/// Error type for avro operations.
pub use error::Error;
/// Configuration for reading avro files.
pub use scan::{Projection, ReadOptions, Reader};
/// Schema retrieval.
pub use schema::get_schema;
/// Incremental avro file writer.
pub use sink::Writer;
