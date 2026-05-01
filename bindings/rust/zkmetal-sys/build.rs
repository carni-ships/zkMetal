fn main() {
    // Only build on aarch64 (Apple Silicon)
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch != "aarch64" {
        println!("cargo:warning=zkmetal-sys only supports aarch64 (Apple Silicon). Skipping C compilation.");
        return;
    }

    let neon_dir: std::path::PathBuf = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("Sources")
        .join("NeonFieldOps");

    let include_dir = neon_dir.join("include");

    // Collect all .c files
    let c_files: Vec<std::path::PathBuf> = std::fs::read_dir(&neon_dir)
        .expect("Failed to read NeonFieldOps source directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "c") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if c_files.is_empty() {
        println!("cargo:warning=No C source files found in {}", neon_dir.display());
    } else {
        let mut build = cc::Build::new();
        build
            .files(&c_files)
            .include(&include_dir)
            .opt_level(3)
            .flag("-march=armv8-a+crypto")
            .flag("-mtune=apple-m1")
            .define("__ARM_NEON", None)
            .warnings(false);

        build.compile("neon_field_ops");
    }

    // Link against zkMetal GPU library
    let zkmetal_lib_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join(".build")
        .join("arm64-apple-macosx")
        .join("release");

    println!("cargo:rustc-link-search=native={}", zkmetal_lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=zkMetal-ffi");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", neon_dir.display());
}
