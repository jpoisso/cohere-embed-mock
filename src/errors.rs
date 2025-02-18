use std::fmt;

pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub(crate) enum Error {
    Embed(String),
    Server(String),
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Server(err) => {
                write!(f, "application server error: '{}'", err)
            }
            Error::Embed(err) => {
                write!(f, "embedding error: '{}'", err)
            }
        }
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for Error {
    fn from(error: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Error::Embed(error.to_string())
    }
}

impl From<hf_hub::api::sync::ApiError> for Error {
    fn from(error: hf_hub::api::sync::ApiError) -> Self {
        Error::Embed(error.to_string())
    }
}

impl From<candle_core::Error> for Error {
    fn from(error: candle_core::Error) -> Self {
        Error::Embed(error.to_string())
    }
}

impl From<std::net::AddrParseError> for Error {
    fn from(error: std::net::AddrParseError) -> Self {
        Error::Embed(error.to_string())
    }
}
