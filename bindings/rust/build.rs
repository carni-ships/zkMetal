use std::env;
use std::path::PathBuf;

fn main() {
    // ----------------------------------------------------------------
    // Resolve the zkMetal library directory.
    //
    // Priority:
    //   1. ZKMETAL_LIB_DIR environment variable (explicit override)
    //   2. Relative path from this crate to the repo .build/release/
    //
    // Build the Swift library first:
    //   cd <zkMetal-repo> && swift build -c release
    //
    // Then build the Rust crate:
    //   ZKMETAL_LIB_DIR=<zkMetal-repo>/.build/release cargo build
    // ----------------------------------------------------------------

    let lib_dir = env::var("ZKMETAL_LIB_DIR").unwrap_or_else(|_| {
        // Navigate from bindings/rust/ -> repo root -> .build/release
        let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
        let mut path = PathBuf::from(&manifest);
        path.pop(); // rust
        path.pop(); // bindings
        path.push(".build");
        path.push("release");
        path.to_string_lossy().to_string()
    });

    println!("cargo:rustc-link-search=native={}", lib_dir);

    // GPU feature: link the Swift-built dynamic library that wraps Metal kernels.
    if cfg!(feature = "gpu") {
        println!("cargo:rustc-link-lib=dylib=zkMetal-ffi");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    // NEON feature: link the static C library for CPU-accelerated field ops.
    if cfg!(feature = "neon") {
        // NeonFieldOps is compiled as a static library by SPM.
        // The .a file is typically at .build/release/libNeonFieldOps.a
        println!("cargo:rustc-link-lib=static=NeonFieldOps");
    }

    println!("cargo:rerun-if-env-changed=ZKMETAL_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");
}
