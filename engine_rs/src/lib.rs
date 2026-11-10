//! engine_rs: the Rust core of the search engine (Phase 3.5).
//!
//! Reads the segments the Python engine writes (engine_py/segment.py) and will run the search
//! loops (daat / wand / bmw) behind one Python call per query.

pub mod delta;
pub mod scoring;
pub mod varint;
