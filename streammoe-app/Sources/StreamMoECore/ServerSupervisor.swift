import Foundation

public enum ServerState: Equatable, Sendable {
    case stopped
    case running(pid: Int32)
    case crashed(stderr: String)
}

public enum ServerError: Error, Equatable {
    case alreadyRunning
    case spawnFailed(String)
}

// Spawner seam. Real implementation fork/execs llama-server; tests swap
// in a fake that just records the intended argv.
public protocol ServerProcessSpawner: Sendable {
    func spawn(path: String, args: [String]) throws -> Int32
    func isAlive(pid: Int32) -> Bool
    func terminate(pid: Int32) throws
}

// Owns the llama-server child process. Builds the argv from the chosen
// ModelSpec + user preferences and mediates start / stop / restart. The
// menu's "Start/Stop StreamMoE" toggle flips this, and also calls
// OllamaController.stop() immediately before starting so the two don't
// race for port 11434.
public final class ServerSupervisor: @unchecked Sendable {
    private let binary: String
    private let spawner: ServerProcessSpawner
    private let network: NetworkAccessSettings
    private let port: Int
    private var _pid: Int32?

    public var state: ServerState {
        guard let pid = _pid else { return .stopped }
        return spawner.isAlive(pid: pid) ? .running(pid: pid) : .stopped
    }

    public init(binary: String,
                spawner: ServerProcessSpawner = PosixSpawner(),
                network: NetworkAccessSettings,
                port: Int = 11434) {
        self.binary = binary
        self.spawner = spawner
        self.network = network
        self.port = port
    }

    public func start(model: ModelSpec) throws {
        if case .running = state { throw ServerError.alreadyRunning }
        var args: [String] = [
            "-m", model.path,
            "--host", network.listenHost,
            "--port", String(port),
            "-ngl", "99",
            "-c", "4096",
            "--mlock",
        ]
        if let sidecar = model.sidecarPath, model.sidecarPresent {
            args += [
                "--moe-sidecar", sidecar,
                "--moe-mode", "slot-bank",
                "--moe-slot-bank", "256",
                "--moe-eager-load",
                "--streammoe-warmup",
            ]
        }
        do {
            _pid = try spawner.spawn(path: binary, args: args)
        } catch {
            throw ServerError.spawnFailed(String(describing: error))
        }
    }

    public func stop() throws {
        guard let pid = _pid else { return }
        try spawner.terminate(pid: pid)
        _pid = nil
    }

    public func restart(model: ModelSpec) throws {
        try stop()
        try start(model: model)
    }
}

// Default concrete spawner. Uses POSIX kill for termination; if the child
// ignores SIGTERM within 3s we escalate to SIGKILL so a wedged model load
// doesn't block the user.
public struct PosixSpawner: ServerProcessSpawner {
    public init() {}

    public func spawn(path: String, args: [String]) throws -> Int32 {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: path)
        task.arguments = args
        // Detach stdio so the child survives if the app quits ungracefully.
        task.standardOutput = FileHandle.nullDevice
        task.standardError = FileHandle.nullDevice
        try task.run()
        return task.processIdentifier
    }

    public func isAlive(pid: Int32) -> Bool {
        // kill -0 returns 0 if signalable, -1 with ESRCH if the pid is gone.
        return kill(pid, 0) == 0
    }

    public func terminate(pid: Int32) throws {
        kill(pid, SIGTERM)
        // Escalate if it didn't exit promptly.
        let deadline = Date().addingTimeInterval(3)
        while isAlive(pid: pid), Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if isAlive(pid: pid) {
            kill(pid, SIGKILL)
        }
    }
}
