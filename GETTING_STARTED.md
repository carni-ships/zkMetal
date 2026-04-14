# Getting Started with zkMetal

A quick guide to getting up and running with zkMetal.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/zkMetal.git
cd zkMetal

# Build (requires macOS 13+ with Metal support)
swift build -c release
```

## Your First MSM

```swift
import zkMetal

// Create an MSM engine
let msm = try MetalMSM()

// Generate random points and scalars for testing
let points = (0..<1024).map { _ in
    PointProjective(
        x: Fr.random(),
        y: Fr.random(),
        z: Fr.one
    )
}
let scalars = (0..<1024).map { _ in Fr.random() }

// Compute MSM
let result = try msm.msm(points: points, scalars: scalars)
```

## Your First NTT

```swift
import zkMetal

let ntt = try NTTEngine()

// Create input values
let values: [Fr] = (0..<1024).map { _ in Fr.random() }

// Forward NTT
let transformed = try ntt.ntt(values)

// Inverse NTT
let restored = try ntt.intt(transformed)
```

## Running Tests

```bash
# Run all tests
swift test

# Run specific test suites
.build/release/zkMetalTests msm
.build/release/zkMetalTests ntt
.build/release/zkMetalTests folding

# List all available tests
.build/release/zkMetalTests --list
```

## Running Benchmarks

```bash
# Run all benchmarks
swift run -c release zkbench all

# Run specific benchmarks
swift run -c release zkbench msm
swift run -c release zkbench ntt
swift run -c release zkbench p2
swift run -c release zkbench fold

# CPU vs GPU comparison
swift run -c release zkbench cpu

# GPU only (skip slow CPU baselines)
swift run -c release zkbench all --no-cpu
```

## Troubleshooting

### "Metal device not found"
- Ensure you're running on macOS with Metal support (M1/M2/M3/M4 chip)
- Check that Xcode Command Line Tools are installed

### "Compilation errors"
- Ensure Swift 5.9+ is installed
- Try cleaning: `rm -rf .build && swift build -c release`

### Tests failing
- Check that GPU is not being throttled
- Some tests require significant GPU memory; close other GPU apps

## Next Steps

- See [README.md](README.md) for all available primitives
- See [PERFORMANCE.md](PERFORMANCE.md) for detailed benchmarks
- See [docs/](docs/) for architecture and tuning guides
- Explore `Sources/zkMetal/` for implementation details
