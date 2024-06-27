use std::fmt::{Display, Formatter, Result};
use anyhow::Error;

// setup error
#[derive(Debug)]
pub struct SetupError(pub &'static str);
impl std::error::Error for SetupError {}
impl Display for SetupError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "Error: {}", self.0)
    }
}

// embedding error
#[derive(Debug)]
pub struct  EmbeddingError;

impl std::error::Error for EmbeddingError {}
impl Display for EmbeddingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "Embedding error")
    }
}
impl From<Error> for EmbeddingError {
    fn from(_:Error) -> Self {
        Self {}
    }
}

// Not available error
#[derive(Debug)]
pub struct NotAvailableError;
impl std::error::Error for NotAvailableError {}
impl Display for NotAvailableError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "File 'not available' error")
    }
}

