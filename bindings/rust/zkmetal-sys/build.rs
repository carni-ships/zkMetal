use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Find the zkMetal root directory (3 levels up from this build.rs)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let zkmetal_root = manifest_dir
        .parent() // rust/
        .unwrap()
        .parent() // bindings/
        .unwrap()
        .parent() // zkMetal root
        .unwrap();

    let lib_dir = zkmetal_root.join("bindings").join("lib");
    let static_lib = lib_dir.join("libzkmetal_ffi.a");

    // Build the FFI library if it doesn't exist
    if !static_lib.exists() {
        let build_script = zkmetal_root.join("bindings").join("build-ffi.sh");
        eprintln!("zkmetal-sys: Building FFI library via {:?}", build_script);
        let status = Command::new("bash")
            .arg(&build_script)
            .current_dir(zkmetal_root)
            .status()
            .expect("Failed to run build-ffi.sh");
        if !status.success() {
            panic!("build-ffi.sh failed with status: {}", status);
        }
    }

    // Tell cargo where to find the library
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=zkmetal_ffi");

    // Swift runtime
    // Find the Swift toolchain lib directory
    let swift_lib = Command::new("xcrun")
        .args(["--toolchain", "default", "--find", "swift"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| {
            PathBuf::from(s.trim())
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .join("lib")
                .join("swift")
                .join("macosx")
        });

    if let Some(swift_lib_dir) = swift_lib {
        if swift_lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", swift_lib_dir.display());
        }
    }

    // Also check Xcode's toolchain path
    if let Ok(output) = Command::new("xcode-select").arg("-p").output() {
        if let Ok(dev_dir) = String::from_utf8(output.stdout) {
            let xcode_swift_lib = PathBuf::from(dev_dir.trim())
                .join("Toolchains")
                .join("XcodeDefault.xctoolchain")
                .join("usr")
                .join("lib")
                .join("swift")
                .join("macosx");
            if xcode_swift_lib.exists() {
                println!(
                    "cargo:rustc-link-search=native={}",
                    xcode_swift_lib.display()
                );
            }
        }
    }

    // macOS frameworks needed by Metal + Swift runtime
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=IOKit");

    // Swift runtime libraries
    println!("cargo:rustc-link-lib=dylib=swiftCore");
    println!("cargo:rustc-link-lib=dylib=swiftFoundation");
    println!("cargo:rustc-link-lib=dylib=swiftMetal");
    println!("cargo:rustc-link-lib=dylib=swiftDarwin");
    println!("cargo:rustc-link-lib=dylib=swiftDispatch");
    println!("cargo:rustc-link-lib=dylib=swiftObjectiveC");

    // Emit the shader directory path for downstream crates
    let shader_dir = zkmetal_root.join("Sources").join("Shaders");
    println!("cargo:shader_dir={}", shader_dir.display());

    // Rerun if the static lib changes
    println!("cargo:rerun-if-changed={}", static_lib.display());
    println!(
        "cargo:rerun-if-changed={}",
        zkmetal_root
            .join("Sources")
            .join("zkMetal-ffi")
            .join("zkMetal_ffi.swift")
            .display()
    );
}
