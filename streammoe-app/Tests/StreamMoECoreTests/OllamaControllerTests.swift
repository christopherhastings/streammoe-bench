import XCTest
@testable import StreamMoECore

// Fake runner that replays canned results in the order they were queued.
// Each call consumes one fixture; unmatched calls fail the test.
final class FakeRunner: ProcessRunner, @unchecked Sendable {
    struct Expectation { let path: String; let args: [String]; let result: ProcessResult }
    private var queue: [Expectation] = []
    private(set) var calls: [(path: String, args: [String])] = []

    func expect(_ path: String, _ args: [String], _ result: ProcessResult) {
        queue.append(.init(path: path, args: args, result: result))
    }

    func run(_ path: String, args: [String], timeout: TimeInterval) throws -> ProcessResult {
        calls.append((path, args))
        guard !queue.isEmpty else {
            XCTFail("unexpected runner call: \(path) \(args)")
            return ProcessResult(exitCode: -1, stdout: "", stderr: "no fixture")
        }
        let next = queue.removeFirst()
        XCTAssertEqual(next.path, path, "command path mismatch")
        XCTAssertEqual(next.args, args, "command args mismatch")
        return next.result
    }
}

final class OllamaControllerTests: XCTestCase {
    func testStateReturnsNotInstalledWhenBinaryMissing() {
        let runner = FakeRunner()
        runner.expect("/usr/bin/which", ["ollama"],
                      ProcessResult(exitCode: 1, stdout: "", stderr: "ollama not found"))
        let sut = OllamaController(runner: runner)
        XCTAssertEqual(sut.state(), .notInstalled)
    }

    func testStateReturnsStoppedWhenBinaryPresentButNoProcess() {
        let runner = FakeRunner()
        runner.expect("/usr/bin/which", ["ollama"],
                      ProcessResult(exitCode: 0, stdout: "/usr/local/bin/ollama\n", stderr: ""))
        runner.expect("/bin/pgrep", ["-xf", "ollama serve"],
                      ProcessResult(exitCode: 1, stdout: "", stderr: ""))
        let sut = OllamaController(runner: runner)
        XCTAssertEqual(sut.state(), .stopped)
    }

    func testStateReturnsRunningWhenPgrepFindsProcess() {
        let runner = FakeRunner()
        runner.expect("/usr/bin/which", ["ollama"],
                      ProcessResult(exitCode: 0, stdout: "/usr/local/bin/ollama\n", stderr: ""))
        runner.expect("/bin/pgrep", ["-xf", "ollama serve"],
                      ProcessResult(exitCode: 0, stdout: "12345\n", stderr: ""))
        let sut = OllamaController(runner: runner)
        XCTAssertEqual(sut.state(), .running(pid: 12345))
    }

    func testStopIssuesLaunchctlBootoutThenFallsBackToKill() throws {
        let runner = FakeRunner()
        // First try the launchd agent — if it's not loaded that's fine.
        runner.expect("/bin/launchctl",
                      ["bootout", "gui/\(getuid())/com.ollama.ollama"],
                      ProcessResult(exitCode: 113, stdout: "", stderr: "Boot-out failed: 113: Could not find specified service"))
        // Then SIGTERM the ollama serve pid.
        runner.expect("/usr/bin/pkill", ["-TERM", "-xf", "ollama serve"],
                      ProcessResult(exitCode: 0, stdout: "", stderr: ""))
        let sut = OllamaController(runner: runner)
        try sut.stop()
        XCTAssertEqual(runner.calls.count, 2)
    }

    func testStartPrefersLaunchctlBootstrapIfAgentFilePresent() throws {
        let runner = FakeRunner()
        runner.expect("/bin/launchctl",
                      ["bootstrap", "gui/\(getuid())",
                       NSHomeDirectory() + "/Library/LaunchAgents/com.ollama.ollama.plist"],
                      ProcessResult(exitCode: 0, stdout: "", stderr: ""))
        let sut = OllamaController(runner: runner,
                                   launchAgentPath: NSHomeDirectory() + "/Library/LaunchAgents/com.ollama.ollama.plist",
                                   launchAgentExists: { _ in true })
        try sut.start()
        XCTAssertEqual(runner.calls.first?.path, "/bin/launchctl")
    }

    func testStartFallsBackToBareOllamaServe() throws {
        let runner = FakeRunner()
        runner.expect("/usr/bin/which", ["ollama"],
                      ProcessResult(exitCode: 0, stdout: "/opt/homebrew/bin/ollama\n", stderr: ""))
        // Expect controller to spawn ollama serve via /bin/sh -c (detached).
        runner.expect("/bin/sh",
                      ["-c", "nohup /opt/homebrew/bin/ollama serve > /dev/null 2>&1 &"],
                      ProcessResult(exitCode: 0, stdout: "", stderr: ""))
        let sut = OllamaController(runner: runner,
                                   launchAgentPath: "/tmp/nope.plist",
                                   launchAgentExists: { _ in false })
        try sut.start()
        XCTAssertEqual(runner.calls.count, 2)
    }
}
