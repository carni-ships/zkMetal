use std::env;

fn main() {
    // The zkmetal-sys crate emits `cargo:shader_dir=...` which becomes
    // DEP_ZKMETAL_FFI_SHADER_DIR for crates that link against it.
    // Re-export it as a compile-time env var for our lib.rs.
    let shader_dir = env::var("DEP_ZKMETAL_FFI_SHADER_DIR").unwrap_or_default();
    println!("cargo:rustc-env=DEP_ZKMETAL_FFI_SHADER_DIR={}", shader_dir);
}
