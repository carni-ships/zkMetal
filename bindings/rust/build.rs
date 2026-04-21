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

    // NEON feature: the NeonFieldOps symbols are embedded in libzkMetal-ffi.dylib.
    // SPM compiles NeonFieldOps as part of the main library, not as a separate .a.
    // No additional linking needed - the zkMetal-ffi dylib already contains these symbols.
    if cfg!(feature = "neon") {
        println!("/* neon symbols are in zkMetal-ffi dylib */");
    }

    println!("cargo:rerun-if-env-changed=ZKMETAL_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");
}
