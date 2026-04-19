import XCTest
@testable import StreamMoECore

// Fake spawner lets tests inspect the exact argv the supervisor would pass
// to llama-server, without actually launching a subprocess.
final class FakeSpawner: ServerProcessSpawner, @unchecked Sendable {
    struct Call { let path: String; let args: [String] }
    private(set) var calls: [Call] = []
    var nextPID: Int32 = 42
    private var lastStopped: Int32 = -1

    func spawn(path: String, args: [String]) throws -> Int32 {
        calls.append(.init(path: path, args: args))
        return nextPID
    }
    func isAlive(pid: Int32) -> Bool { pid != lastStopped }
    func terminate(pid: Int32) throws { lastStopped = pid }
}

final class ServerSupervisorTests: XCTestCase {
    private func makeSpec(id: String = "qwen-35b", sidecar: String? = "/tmp/qwen-35b-sidecar") -> ModelSpec {
        ModelSpec(id: id, path: "/tmp/\(id).gguf", sizeBytes: 100,
                  sidecarPresent: sidecar != nil, sidecarPath: sidecar)
    }

    func testStartPassesAllMandatoryFlags() throws {
        let spawner = FakeSpawner()
        let net = NetworkAccessSettings(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let sut = ServerSupervisor(binary: "/tmp/llama-server", spawner: spawner, network: net)

        try sut.start(model: makeSpec())

        let args = spawner.calls.first!.args
        XCTAssertTrue(args.contains("-m"))
        XCTAssertTrue(args.contains("/tmp/qwen-35b.gguf"))
        XCTAssertTrue(args.contains("--moe-sidecar"))
        XCTAssertTrue(args.contains("/tmp/qwen-35b-sidecar"))
        XCTAssertTrue(args.contains("--moe-eager-load"))
        XCTAssertTrue(args.contains("--streammoe-warmup"))
        XCTAssertTrue(args.contains("--host"))
        XCTAssertEqual(sut.state, .running(pid: 42))
    }

    func testStartUsesLoopbackByDefault() throws {
        let spawner = FakeSpawner()
        let net = NetworkAccessSettings(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let sut = ServerSupervisor(binary: "/tmp/llama-server", spawner: spawner, network: net)
        try sut.start(model: makeSpec())
        let i = spawner.calls.first!.args.firstIndex(of: "--host")!
        XCTAssertEqual(spawner.calls.first!.args[i + 1], "127.0.0.1")
    }

    func testStartUsesWildcardWhenLANAllowed() throws {
        let spawner = FakeSpawner()
        let defaults = UserDefaults(suiteName: UUID().uuidString)!
        let net = NetworkAccessSettings(defaults: defaults)
        net.allowLAN = true
        let sut = ServerSupervisor(binary: "/tmp/llama-server", spawner: spawner, network: net)
        try sut.start(model: makeSpec())
        let i = spawner.calls.first!.args.firstIndex(of: "--host")!
        XCTAssertEqual(spawner.calls.first!.args[i + 1], "0.0.0.0")
    }

    func testStartOmitsSidecarFlagsIfNotPresent() throws {
        let spawner = FakeSpawner()
        let net = NetworkAccessSettings(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let sut = ServerSupervisor(binary: "/tmp/llama-server", spawner: spawner, network: net)
        try sut.start(model: makeSpec(sidecar: nil))
        let args = spawner.calls.first!.args
        XCTAssertFalse(args.contains("--moe-sidecar"))
        XCTAssertFalse(args.contains("--moe-eager-load"))
    }

    func testStopTerminatesTheSpawnedProcess() throws {
        let spawner = FakeSpawner()
        let net = NetworkAccessSettings(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let sut = ServerSupervisor(binary: "/tmp/llama-server", spawner: spawner, network: net)
        try sut.start(model: makeSpec())
        XCTAssertEqual(sut.state, .running(pid: 42))
        try sut.stop()
        XCTAssertEqual(sut.state, .stopped)
    }

    func testRestartStopsThenStartsWithNewModel() throws {
        let spawner = FakeSpawner()
        let net = NetworkAccessSettings(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let sut = ServerSupervisor(binary: "/tmp/llama-server", spawner: spawner, network: net)
        try sut.start(model: makeSpec(id: "a"))
        spawner.nextPID = 43
        try sut.restart(model: makeSpec(id: "b"))
        XCTAssertEqual(spawner.calls.count, 2)
        XCTAssertTrue(spawner.calls.last!.args.contains("/tmp/b.gguf"))
        XCTAssertEqual(sut.state, .running(pid: 43))
    }

    func testStartRejectsWhenAlreadyRunning() throws {
        let spawner = FakeSpawner()
        let net = NetworkAccessSettings(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let sut = ServerSupervisor(binary: "/tmp/llama-server", spawner: spawner, network: net)
        try sut.start(model: makeSpec(id: "a"))
        XCTAssertThrowsError(try sut.start(model: makeSpec(id: "b")))
    }
}
