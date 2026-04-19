import Foundation

// Shell-out helpers to figure out which client apps the user has on this Mac.
// Kept intentionally dumb — each detector is a single process invocation with
// a short timeout. Cached for 30s so opening the same pane repeatedly doesn't
// re-shell. All detectors return a concrete boolean; "unknown" is modeled as
// `false` to avoid hiding panes that might still work fine without detection.
final class AppDetector {
    struct Cache {
        var timestamp: Date
        var openWebUIInDocker: Bool
        var openWebUINative: Bool
        var cursorInstalled: Bool
        var lmStudioInstalled: Bool
        var claudeDesktopInstalled: Bool
    }

    private var cache: Cache? = nil
    private let ttl: TimeInterval = 30

    func snapshot() -> Cache {
        if let c = cache, Date().timeIntervalSince(c.timestamp) < ttl { return c }
        let c = Cache(
            timestamp: Date(),
            openWebUIInDocker: Self.shellHasOutput("docker ps --format '{{.Image}}' 2>/dev/null | grep -i open-webui"),
            openWebUINative:   Self.shellHasOutput("pgrep -f 'open-webui' 2>/dev/null"),
            cursorInstalled:   FileManager.default.fileExists(atPath: "/Applications/Cursor.app"),
            lmStudioInstalled: FileManager.default.fileExists(atPath: "/Applications/LM Studio.app"),
            claudeDesktopInstalled: FileManager.default.fileExists(atPath: "/Applications/Claude.app")
        )
        cache = c
        return c
    }

    // Runs a /bin/sh -c command with a 2s timeout, returning true if stdout
    // is non-empty. Suppresses all errors — a missing binary (e.g. no docker
    // installed) should just read as "nothing found."
    private static func shellHasOutput(_ cmd: String) -> Bool {
        let task = Process()
        task.launchPath = "/bin/sh"
        task.arguments  = ["-c", cmd]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = Pipe()
        do { try task.run() } catch { return false }
        let deadline = Date().addingTimeInterval(2)
        while task.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if task.isRunning { task.terminate(); return false }
        let data = pipe.fileHandleForReading.availableData
        return !data.isEmpty
    }
}
