use std::env;
use std::path::PathBuf;

fn main() {
    // Link against the zkMetal-ffi dynamic library built by Swift Package Manager.
    //
    // Users must set ZKMETAL_LIB_DIR to the directory containing libzkMetal-ffi.dylib,
    // typically: <zkMetal-repo>/.build/release/
    //
    // Build the Swift library first:
    //   cd <zkMetal-repo> && swift build -c release
    //
    // Then build the Rust crate:
    //   ZKMETAL_LIB_DIR=<zkMetal-repo>/.build/release cargo build

    let lib_dir = env::var("ZKMETAL_LIB_DIR").unwrap_or_else(|_| {
        // Try to find relative to manifest dir (typical development layout)
        let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
        let mut path = PathBuf::from(&manifest);
        // Navigate from Sources/zkMetal-ffi/RustBindings/ -> repo root -> .build/release
        path.pop(); // RustBindings
        path.pop(); // zkMetal-ffi
        path.pop(); // Sources
        path.push(".build");
        path.push("release");
        path.to_string_lossy().to_string()
    });

    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=dylib=zkMetal-ffi");

    // Also link Apple frameworks that the Swift library depends on
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    // Re-run if the lib changes
    println!("cargo:rerun-if-env-changed=ZKMETAL_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");
}
