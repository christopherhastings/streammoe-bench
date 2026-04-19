import Foundation

public enum OllamaState: Equatable, Sendable {
    case notInstalled
    case stopped
    case running(pid: Int)
}

public enum OllamaError: Error, Equatable {
    case notInstalled
    case stopFailed(stderr: String)
    case startFailed(stderr: String)
}

// Owns the "turn Ollama off when StreamMoE is ON" toggle. StreamMoE and Ollama
// both want port 11434, so they can't run simultaneously — we flip Ollama off
// before binding llama-server, and offer to flip it back on when the user
// disables StreamMoE.
//
// Strategy is defensive: Ollama's install story varies (Homebrew, the
// official .pkg, or a manual launch). We therefore try the most reliable
// launchd route first, and fall back to pkill / nohup for manual installs.
public final class OllamaController: @unchecked Sendable {
    private let runner: ProcessRunner
    private let launchAgentPath: String
    private let launchAgentExists: (String) -> Bool

    public init(
        runner: ProcessRunner = SystemProcessRunner(),
        launchAgentPath: String = NSHomeDirectory() + "/Library/LaunchAgents/com.ollama.ollama.plist",
        launchAgentExists: @escaping (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
    ) {
        self.runner = runner
        self.launchAgentPath = launchAgentPath
        self.launchAgentExists = launchAgentExists
    }

    public func state() -> OllamaState {
        // (1) Is the ollama binary on PATH at all?
        let which = (try? runner.run("/usr/bin/which", args: ["ollama"], timeout: 2))
            ?? ProcessResult(exitCode: -1, stdout: "", stderr: "")
        if which.exitCode != 0 || which.stdout.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return .notInstalled
        }

        // (2) Is `ollama serve` actually running?
        let pg = (try? runner.run("/bin/pgrep", args: ["-xf", "ollama serve"], timeout: 2))
            ?? ProcessResult(exitCode: -1, stdout: "", stderr: "")
        if pg.exitCode == 0, let pid = Int(pg.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: "\n").first.map(String.init) ?? "") {
            return .running(pid: pid)
        }
        return .stopped
    }

    public func stop() throws {
        // Try the launchd agent first — a bootout reliably terminates the
        // managed process even if it's been restarted by KeepAlive.
        _ = try? runner.run("/bin/launchctl",
                            args: ["bootout", "gui/\(getuid())/com.ollama.ollama"],
                            timeout: 5)
        // Always follow up with pkill for manual installs and as a guarantee.
        let kill = try runner.run("/usr/bin/pkill",
                                  args: ["-TERM", "-xf", "ollama serve"],
                                  timeout: 3)
        if kill.exitCode != 0 && kill.exitCode != 1 {
            // pkill returns 1 when no process matched — that's fine (already
            // stopped); any other non-zero exit is an actual failure.
            throw OllamaError.stopFailed(stderr: kill.stderr)
        }
    }

    public func start() throws {
        // Preferred path: user has a LaunchAgent installed — bootstrap it.
        if launchAgentExists(launchAgentPath) {
            let boot = try runner.run("/bin/launchctl",
                                      args: ["bootstrap", "gui/\(getuid())", launchAgentPath],
                                      timeout: 5)
            if boot.exitCode == 0 { return }
            // Fall through to direct-spawn if bootstrap failed.
        }

        // Fallback: spawn `ollama serve` directly, detached.
        let which = try runner.run("/usr/bin/which", args: ["ollama"], timeout: 2)
        let bin = which.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard which.exitCode == 0, !bin.isEmpty else {
            throw OllamaError.notInstalled
        }
        let spawn = try runner.run("/bin/sh",
                                   args: ["-c", "nohup \(bin) serve > /dev/null 2>&1 &"],
                                   timeout: 5)
        if spawn.exitCode != 0 {
            throw OllamaError.startFailed(stderr: spawn.stderr)
        }
    }
}
