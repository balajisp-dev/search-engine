//! BM25 formula pieces: the Rust twin of engine_py/scoring.py and the L / tf part in segment.py.
//!
//! ```text
//! IDF(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
//! L        = 1 - b + b * dl / avgdl
//! TF-part  = tf * (K1 + 1) / (tf + K1 * L)
//! ```
//!
//! Must match Python to ~1e-12, because the stored bounds (ub, block_max) were computed in Python.

/// BM25 parameters (same as engine_py/segment.py).
pub const K1: f64 = 1.5;
pub const B: f64 = 0.75;

/// Lucene BM25 IDF. `n` = number of docs, `df` = docs containing the term.
pub fn idf(n: u32, df: u32) -> f64 {
    (1.0 + (n as f64 - df as f64 + 0.5) / (df as f64 + 0.5)).ln()
}

/// Length normalisation for a doc of `doc_len` tokens.
pub fn length_norm(doc_len: u32, avgdl: f64) -> f64 {
    1.0 - B + B * (doc_len as f64) / avgdl
}

/// The TF part for a term appearing `tf` times in a doc whose length_norm is `l`.
pub fn tf_part(tf: u32, l: f64) -> f64 {
    ((K1 + 1.0) * tf as f64) / (tf as f64 + K1 * l)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Floats are compared with a tolerance, like pytest.approx.
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    // Expected values below were printed by the Python code (engine_py), so Rust must agree.
    const SCIFACT_N: u32 = 5183;
    const SCIFACT_AVGDL: f64 = 156.68030098398611;

    #[test]
    fn idf_matches_python() {
        assert!(close(idf(SCIFACT_N, 1), 8.147867129923947)); // rarest term
        assert!(close(idf(SCIFACT_N, 2530), 0.7171600473779779)); // "cell"
        assert!(close(idf(SCIFACT_N, SCIFACT_N), 9.64552689437707e-05)); // in every doc: tiny, never negative
        assert!(close(idf(SCIFACT_N, 0), 9.246479418592056)); // unseen term: finite
    }

    #[test]
    fn length_norm_matches_python() {
        assert!(close(length_norm(120, SCIFACT_AVGDL), 0.8244180949026813)); // shorter than average: < 1
        assert!(close(length_norm(400, SCIFACT_AVGDL), 2.1647269830089373)); // longer than average: > 1
    }

    #[test]
    fn tf_part_matches_python() {
        let l = length_norm(120, SCIFACT_AVGDL);
        assert!(close(tf_part(3, l), 1.7702761531742277));
    }

    #[test]
    fn tf_part_saturates() {
        // more occurrences always help, but never past K1 + 1
        assert!(tf_part(2, 1.0) > tf_part(1, 1.0));
        assert!(tf_part(1_000_000, 1.0) < K1 + 1.0);
    }
}
