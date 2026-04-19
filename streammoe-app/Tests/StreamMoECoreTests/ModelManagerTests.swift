import XCTest
@testable import StreamMoECore

final class ModelManagerTests: XCTestCase {
    private var tempDir: URL!

    override func setUp() async throws {
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mm-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    private func write(_ name: String, bytes: Int) throws {
        let url = tempDir.appendingPathComponent(name)
        let data = Data(count: bytes)
        try data.write(to: url)
    }

    func testListLocalEmptyOnEmptyDir() {
        let sut = ModelManager(modelsDir: tempDir)
        XCTAssertTrue(sut.listLocal().isEmpty)
    }

    func testListLocalFindsGGUFs() throws {
        try write("qwen3-0.6b.gguf", bytes: 100)
        try write("qwen3-35b-q4.gguf", bytes: 200)
        try write("NOTAMODEL.txt", bytes: 10)
        let sut = ModelManager(modelsDir: tempDir)
        let models = sut.listLocal().map(\.id).sorted()
        XCTAssertEqual(models, ["qwen3-0.6b", "qwen3-35b-q4"])
    }

    func testListLocalFlagsModelsNeedingSidecar() throws {
        // Convention: if a sibling directory ends in "-sidecar" and contains
        // a manifest.json, mark the model as "sidecar present" so the UI
        // can show a Prepare / Verify badge.
        try write("qwen3-35b-q4.gguf", bytes: 200)
        let sidecarDir = tempDir.appendingPathComponent("qwen3-35b-q4-sidecar")
        try FileManager.default.createDirectory(at: sidecarDir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: sidecarDir.appendingPathComponent("manifest.json"))
        let sut = ModelManager(modelsDir: tempDir)
        let spec = sut.listLocal().first { $0.id == "qwen3-35b-q4" }!
        XCTAssertTrue(spec.sidecarPresent)
        XCTAssertEqual(spec.sidecarPath, sidecarDir.path)
    }

    func testCurrentModelIDFromSnapshotMatchesListed() throws {
        try write("qwen3-35b-q4.gguf", bytes: 200)
        let sut = ModelManager(modelsDir: tempDir)
        var snap = ServerStatusSnapshot()
        snap.id = "qwen3-35b-q4"
        XCTAssertEqual(sut.currentModel(from: snap)?.id, "qwen3-35b-q4")
    }
}
