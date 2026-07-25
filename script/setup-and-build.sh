#!/bin/bash
set -e

echo "=== Installing Rust toolchain ==="
if ! command -v rustc &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo "=== Installing Xcode Metal toolchain ==="
xcodebuild -downloadComponent MetalToolchain 2>/dev/null || echo "Metal toolchain already installed or download initiated"

echo "=== Installing cargo-bundle ==="
if ! command -v cargo-bundle &>/dev/null; then
    cargo install cargo-bundle
fi

echo "=== Installing jq ==="
if ! command -v jq &>/dev/null; then
    brew install jq
fi

echo "=== Running bootstrap ==="
./script/bootstrap --skip-common-skills

echo "=== Building and running Warp ==="
cd app && cargo run --bin warp-oss --features gui
