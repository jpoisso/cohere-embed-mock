##### Rust Builder
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies
RUN apt-get update && apt-get upgrade -yq \
    && apt-get install curl build-essential cmake musl-tools musl-dev -yq \
    && apt-get clean
RUN ln -s "/usr/bin/g++" "/usr/bin/musl-g++"

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Create fresh workspace
WORKDIR /opt
RUN cargo new cohere-embed-mock

# Cache dependencies by themselves
COPY Cargo.toml Cargo.lock /opt/cohere-embed-mock/
WORKDIR /opt/cohere-embed-mock
RUN rustup target add x86_64-unknown-linux-musl
RUN cargo build --target x86_64-unknown-linux-musl --release

# Build the application
COPY src /opt/cohere-embed-mock/src/
RUN touch /opt/cohere-embed-mock/src/main.rs
RUN cargo build --target x86_64-unknown-linux-musl --release

# Define a user
RUN useradd -u 10001 appuser

##### Runtime
FROM scratch

# Copy generated binary and user in runtime
COPY --from=builder /opt/cohere-embed-mock/target/x86_64-unknown-linux-musl/release/cohere-embed-mock /app
COPY --from=builder /etc/passwd /etc/passwd

# Run as regular user (not root)
USER 10001

ENTRYPOINT ["/app"]