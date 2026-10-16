import CoreML
import Foundation
import os

/// Diagnostics: prints, for each bundled model, how Core ML places its operations
/// on the Neural Engine / GPU / CPU (share of Core ML's estimated cost), using
/// MLComputePlan. Output goes to Xcode's console, prefixed "[FaceTrack] plan".
enum ComputePlanReport {
    private static let store = OSAllocatedUnfairLock(initialState: [String]())
    /// Summary lines collected so far (re-printed periodically by Pipeline).
    static var lines: [String] { store.withLock { $0 } }

    private static func emit(_ line: String) {
        print(line)
        store.withLock { $0.append(line) }
    }
    static func printAll() {
        let jobs: [(String, String?, MLComputeUnits)] = [
            ("YuNet", nil, .cpuAndGPU),
            ("EdgeTAMImageEncoder", nil, .cpuAndNeuralEngine),
            ("EdgeTAMMemoryAttention", nil, .cpuAndNeuralEngine),
            ("EdgeTAMMaskDecoder", "track", .cpuAndNeuralEngine),
            ("EdgeTAMMemoryEncoder", nil, .cpuAndNeuralEngine),
        ]
        Task.detached(priority: .utility) {
            for (name, function, units) in jobs {
                await report(name, function: function, units: units)
            }
        }
    }

    private static func report(_ name: String, function: String?, units: MLComputeUnits) async {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { return }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        if let function { cfg.functionName = function }
        do {
            let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
            guard case .program(let program) = plan.modelStructure,
                  let fn = program.functions[function ?? "main"] ?? program.functions.values.first else {
                emit("[FaceTrack] plan \(name): not an ML program")
                return
            }
            var cost: [String: Double] = [:]
            var count: [String: Int] = [:]
            var cpuOps: [String: Int] = [:]
            func visit(_ block: MLModelStructure.Program.Block) {
                for op in block.operations {
                    for b in op.blocks { visit(b) }
                    guard let usage = plan.deviceUsage(for: op) else { continue }  // consts etc.
                    let dev: String
                    switch usage.preferred {
                    case .neuralEngine: dev = "ANE"
                    case .gpu: dev = "GPU"
                    case .cpu: dev = "CPU"
                    @unknown default: dev = "other"
                    }
                    count[dev, default: 0] += 1
                    cost[dev, default: 0] += plan.estimatedCost(of: op)?.weight ?? 0
                    if dev != "ANE" { cpuOps[op.operatorName, default: 0] += 1 }
                }
            }
            visit(fn.block)
            let total = max(cost.values.reduce(0, +), 1e-9)
            let share = ["ANE", "GPU", "CPU"].map { String(format: "%@ %.0f%% (%d ops)", $0, 100 * (cost[$0] ?? 0) / total, count[$0] ?? 0) }
            let offANE = cpuOps.sorted { $0.value > $1.value }.prefix(6).map { "\($0.key)×\($0.value)" }.joined(separator: ", ")
            emit("[FaceTrack] plan \(name)\(function.map { ":" + $0 } ?? ""): " + share.joined(separator: " | ")
                  + (offANE.isEmpty ? "" : " | off-ANE ops: " + offANE))
        } catch {
            emit("[FaceTrack] plan \(name): \(error.localizedDescription)")
        }
    }
}
