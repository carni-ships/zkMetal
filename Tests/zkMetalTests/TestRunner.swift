// Lightweight test framework — works without Xcode/XCTest (CommandLineTools only)
// Run with: swift build && .build/debug/zkMetalTests
// Or: swift build -c release && .build/release/zkMetalTests

import Foundation

// MARK: - Minimal test framework

var _passed = 0
var _failed = 0
var _currentSuite = ""

func suite(_ name: String) {
    _currentSuite = name
    print("\n--- \(name) ---")
}

func expect(_ condition: Bool, _ msg: String = "", file: String = #file, line: Int = #line) {
    if condition {
        _passed += 1
    } else {
        let loc = URL(fileURLWithPath: file).lastPathComponent
        print("  [FAIL] \(loc):\(line) \(msg)")
        _failed += 1
    }
}

func expectEqual<T: Equatable>(_ a: T, _ b: T, _ msg: String = "", file: String = #file, line: Int = #line) {
    if a == b {
        _passed += 1
    } else {
        let loc = URL(fileURLWithPath: file).lastPathComponent
        print("  [FAIL] \(loc):\(line) \(msg): got \(a), expected \(b)")
        _failed += 1
    }
}

func expectThrows<T>(_ expr: @autoclosure () throws -> T, _ msg: String = "", file: String = #file, line: Int = #line) {
    do {
        _ = try expr()
        let loc = URL(fileURLWithPath: file).lastPathComponent
        print("  [FAIL] \(loc):\(line) expected throw: \(msg)")
        _failed += 1
    } catch {
        _passed += 1
    }
}

func printSummary() {
    let total = _passed + _failed
    print(String(format: "\n=== Test Summary: %d passed, %d failed (%d total) ===", _passed, _failed, total))
    if _failed > 0 { print("*** SOME TESTS FAILED ***") }
    fflush(stdout); fflush(stderr)
    exit(_failed > 0 ? 1 : 0)
}
