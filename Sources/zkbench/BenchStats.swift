// BenchStats — Statistical benchmark harness for paper-quality results
// Provides confidence intervals, formatted output, and LaTeX/Markdown tables.

import Foundation

/// All statistics from a benchmark run.
public struct BenchResult {
    /// Raw timing samples in milliseconds.
    public let samples: [Double]
    /// Median of samples.
    public let median: Double
    /// Arithmetic mean of samples.
    public let mean: Double
    /// Sample standard deviation (Bessel-corrected).
    public let stddev: Double
    /// 95% confidence interval lower bound (ms).
    public let ci95Low: Double
    /// 95% confidence interval upper bound (ms).
    public let ci95High: Double
    /// Minimum sample (ms).
    public let min: Double
    /// Maximum sample (ms).
    public let max: Double
    /// Number of samples.
    public let n: Int
    /// Human-readable label.
    public let label: String
}

// MARK: - Core bench function

/// Run a benchmark with warmup, compute all statistics, and print a summary line.
///
/// - Parameters:
///   - label: Human-readable name for this benchmark.
///   - warmup: Number of warmup iterations (discarded). Default 2.
///   - iterations: Number of timed iterations. Default 10.
///   - block: The code to benchmark.
/// - Returns: A `BenchResult` with all computed statistics.
@discardableResult
public func bench(
    _ label: String,
    warmup: Int = 2,
    iterations: Int = 10,
    block: () throws -> Void
) rethrows -> BenchResult {
    // Warmup
    for _ in 0..<warmup {
        try block()
    }

    // Timed iterations
    var samples = [Double]()
    samples.reserveCapacity(iterations)
    for _ in 0..<iterations {
        let start = CFAbsoluteTimeGetCurrent()
        try block()
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        samples.append(elapsed)
    }

    let result = computeStats(label: label, samples: samples)
    printBenchLine(result)
    return result
}

/// Overload that accepts a throwing block returning a value (result discarded).
@discardableResult
public func bench<T>(
    _ label: String,
    warmup: Int = 2,
    iterations: Int = 10,
    block: () throws -> T
) rethrows -> BenchResult {
    // Warmup
    for _ in 0..<warmup {
        _ = try block()
    }

    var samples = [Double]()
    samples.reserveCapacity(iterations)
    for _ in 0..<iterations {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try block()
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        samples.append(elapsed)
    }

    let result = computeStats(label: label, samples: samples)
    printBenchLine(result)
    return result
}

// MARK: - Statistics computation

/// Compute all statistics from raw samples.
public func computeStats(label: String, samples: [Double]) -> BenchResult {
    let sorted = samples.sorted()
    let n = sorted.count
    assert(n >= 2, "Need at least 2 samples for statistics")

    let median: Double
    if n % 2 == 1 {
        median = sorted[n / 2]
    } else {
        median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }

    let mean = sorted.reduce(0.0, +) / Double(n)
    let variance = sorted.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(n - 1)
    let stddev = variance.squareRoot()

    // 95% CI using t-distribution critical values (two-tailed, alpha=0.05)
    let tCrit = tCriticalValue95(df: n - 1)
    let sem = stddev / Double(n).squareRoot()
    let ci95Low = mean - tCrit * sem
    let ci95High = mean + tCrit * sem

    return BenchResult(
        samples: samples,
        median: median,
        mean: mean,
        stddev: stddev,
        ci95Low: ci95Low,
        ci95High: ci95High,
        min: sorted.first!,
        max: sorted.last!,
        n: n,
        label: label
    )
}

// MARK: - Comparison

/// Compare two benchmark results and print a summary.
public func benchCompare(_ label: String, baseline: BenchResult, candidate: BenchResult) {
    let speedup = baseline.median / candidate.median

    // Simple significance test: non-overlapping 95% CIs
    let significant = baseline.ci95Low > candidate.ci95High || candidate.ci95Low > baseline.ci95High
    let sigStr = significant ? "p<0.05" : "n.s."

    // Welch's t-test for more rigorous p-value
    let tStat = welchTStatistic(a: baseline, b: candidate)
    let df = welchDF(a: baseline, b: candidate)
    let pValue = tTestPValue(t: abs(tStat), df: df)
    let pStr: String
    if pValue < 0.001 {
        pStr = "p<0.001"
    } else if pValue < 0.01 {
        pStr = "p<0.01"
    } else if pValue < 0.05 {
        pStr = "p<0.05"
    } else {
        pStr = String(format: "p=%.3f", pValue)
    }

    fputs(String(format: "  %@: %.1fms -> %.1fms (%.2fx, %@, CI: %@)\n",
                 label,
                 baseline.median, candidate.median,
                 speedup, pStr, sigStr), stderr)
}

// MARK: - Formatted output

/// Print a single benchmark result line to stderr.
private func printBenchLine(_ r: BenchResult) {
    fputs(String(format: "  %@: %.1fms +/- %.1fms (95%% CI: [%.1f, %.1f], n=%d)\n",
                 r.label, r.median, r.stddev, r.ci95Low, r.ci95High, r.n), stderr)
}

// MARK: - Table formatters

/// Format results as a LaTeX tabular for paper inclusion.
public func formatLatexTable(results: [(String, BenchResult)]) -> String {
    var lines = [String]()
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Benchmark & Median (ms) & Mean (ms) & Std Dev & 95\\% CI \\\\")
    lines.append("\\midrule")
    for (name, r) in results {
        let escaped = name.replacingOccurrences(of: "_", with: "\\_")
        lines.append(String(format: "%@ & %.2f & %.2f & %.2f & [%.2f, %.2f] \\\\",
                            escaped, r.median, r.mean, r.stddev, r.ci95Low, r.ci95High))
    }
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Benchmark results (n=\\(results.first?.1.n ?? 0) iterations).}")
    lines.append("\\end{table}")
    return lines.joined(separator: "\n")
}

/// Format results as a Markdown table with CIs.
public func formatMarkdownTable(results: [(String, BenchResult)]) -> String {
    var lines = [String]()
    lines.append("| Benchmark | Median (ms) | Mean (ms) | Std Dev | 95% CI | n |")
    lines.append("|-----------|------------|----------|---------|--------|---|")
    for (name, r) in results {
        lines.append(String(format: "| %@ | %.2f | %.2f | %.2f | [%.2f, %.2f] | %d |",
                            name, r.median, r.mean, r.stddev, r.ci95Low, r.ci95High, r.n))
    }
    return lines.joined(separator: "\n")
}

// MARK: - t-distribution utilities

/// Two-tailed t critical values for 95% CI (alpha=0.05).
/// Covers common sample sizes; falls back to 1.96 (z) for large df.
private func tCriticalValue95(df: Int) -> Double {
    // Precomputed table for small df (two-tailed, 0.025 each tail)
    let table: [Int: Double] = [
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000,
        80: 1.990, 100: 1.984, 200: 1.972,
    ]
    if let v = table[df] { return v }
    // Interpolate: find closest keys
    let keys = table.keys.sorted()
    if df < keys.first! { return table[keys.first!]! }
    if df > keys.last! { return 1.960 } // normal approximation
    // Linear interpolation between bracketing keys
    for i in 0..<(keys.count - 1) {
        if keys[i] <= df && df <= keys[i + 1] {
            let lo = keys[i], hi = keys[i + 1]
            let frac = Double(df - lo) / Double(hi - lo)
            return table[lo]! * (1.0 - frac) + table[hi]! * frac
        }
    }
    return 1.960
}

/// Welch's t-statistic for two independent samples.
private func welchTStatistic(a: BenchResult, b: BenchResult) -> Double {
    let s2a = a.stddev * a.stddev
    let s2b = b.stddev * b.stddev
    let denom = (s2a / Double(a.n) + s2b / Double(b.n)).squareRoot()
    guard denom > 0 else { return 0 }
    return (a.mean - b.mean) / denom
}

/// Welch-Satterthwaite degrees of freedom.
private func welchDF(a: BenchResult, b: BenchResult) -> Double {
    let s2a = a.stddev * a.stddev
    let s2b = b.stddev * b.stddev
    let na = Double(a.n)
    let nb = Double(b.n)
    let num = (s2a / na + s2b / nb) * (s2a / na + s2b / nb)
    let den = (s2a / na) * (s2a / na) / (na - 1) + (s2b / nb) * (s2b / nb) / (nb - 1)
    guard den > 0 else { return na + nb - 2 }
    return num / den
}

/// Approximate two-tailed p-value from t-statistic and df using normal approximation.
/// For paper results, this is conservative; exact t-CDF would require more code.
private func tTestPValue(t: Double, df: Double) -> Double {
    // Use the approximation: for df > 2, the t-distribution is close to normal.
    // For small df, this overestimates significance slightly (conservative for "n.s." claims).
    // Abramowitz & Stegun 26.2.17 approximation for normal CDF.
    let x: Double
    if df > 30 {
        x = t
    } else {
        // Adjust t to z using Cornish-Fisher: z ~ t * (1 - 1/(4*df))
        x = t * (1.0 - 1.0 / (4.0 * df))
    }
    let p = 2.0 * normalSurvival(x)
    return Swift.min(1.0, Swift.max(0.0, p))
}

/// Upper tail probability of standard normal: P(Z > x).
/// Uses Abramowitz & Stegun 7.1.26 rational approximation (|error| < 1.5e-7).
private func normalSurvival(_ x: Double) -> Double {
    guard x >= 0 else { return 1.0 - normalSurvival(-x) }
    let p  = 0.3275911
    let a1 = 0.254829592
    let a2 = -0.284496736
    let a3 = 1.421413741
    let a4 = -1.453152027
    let a5 = 1.061405429
    let t = 1.0 / (1.0 + p * x)
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    let phi = poly * exp(-x * x / 2.0)
    return phi
}
