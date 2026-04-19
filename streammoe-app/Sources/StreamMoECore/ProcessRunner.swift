import Foundation

// Seams for shelling out. The production implementation calls Process; tests
// inject a fake. Using a protocol rather than @testable import keeps the test
// target free of Process/Pipe lifecycle issues on CI.
public protocol ProcessRunner: Sendable {
    func run(_ path: String, args: [String], timeout: TimeInterval) throws -> ProcessResult
}

public struct ProcessResult: Equatable, Sendable {
    public let exitCode: Int32
    public let stdout: String
    public let stderr: String

    public init(exitCode: Int32, stdout: String, stderr: String) {
        self.exitCode = exitCode
        self.stdout = stdout
        self.stderr = stderr
    }
}

public struct SystemProcessRunner: ProcessRunner {
    public init() {}

    public func run(_ path: String, args: [String], timeout: TimeInterval = 5) throws -> ProcessResult {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: path)
        task.arguments = args
        let out = Pipe(), err = Pipe()
        task.standardOutput = out
        task.standardError = err
        try task.run()

        // Bounded wait. If the child exceeds the timeout, terminate rather
        // than block the caller's menu thread. The caller sees exitCode = -1.
        let deadline = Date().addingTimeInterval(timeout)
        while task.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.02)
        }
        if task.isRunning {
            task.terminate()
            return ProcessResult(exitCode: -1, stdout: "", stderr: "timeout")
        }
        let so = String(data: out.fileHandleForReading.availableData, encoding: .utf8) ?? ""
        let se = String(data: err.fileHandleForReading.availableData, encoding: .utf8) ?? ""
        return ProcessResult(exitCode: task.terminationStatus, stdout: so, stderr: se)
    }
}
