import Foundation

public struct ModelSpec: Equatable, Sendable {
    public let id: String
    public let path: String
    public let sizeBytes: Int64
    public let sidecarPresent: Bool
    public let sidecarPath: String?
}

// Enumerates model files on disk and cross-references with /streammoe/status
// to identify which is currently loaded. This is what the menu-bar model
// dropdown shows; it's also what the Welcome wizard's "pick a model" screen
// lists. Intentionally read-only — download/prepare lifecycle is a separate
// ModelDownloader concern that can be added once we pick a source.
public final class ModelManager: @unchecked Sendable {
    private let modelsDir: URL
    private let fm: FileManager

    public init(modelsDir: URL, fileManager: FileManager = .default) {
        self.modelsDir = modelsDir
        self.fm = fileManager
    }

    public func listLocal() -> [ModelSpec] {
        guard let entries = try? fm.contentsOfDirectory(atPath: modelsDir.path) else { return [] }
        var result: [ModelSpec] = []
        for entry in entries where entry.hasSuffix(".gguf") {
            let path = modelsDir.appendingPathComponent(entry).path
            guard let attrs = try? fm.attributesOfItem(atPath: path) else { continue }
            let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
            let id = (entry as NSString).deletingPathExtension
            let sidecarDir = modelsDir.appendingPathComponent("\(id)-sidecar")
            let manifest = sidecarDir.appendingPathComponent("manifest.json")
            let sidecarPresent = fm.fileExists(atPath: manifest.path)
            result.append(ModelSpec(
                id: id, path: path, sizeBytes: size,
                sidecarPresent: sidecarPresent,
                sidecarPath: sidecarPresent ? sidecarDir.path : nil
            ))
        }
        return result.sorted { $0.id < $1.id }
    }

    public func currentModel(from snap: ServerStatusSnapshot) -> ModelSpec? {
        listLocal().first { $0.id == snap.id }
    }
}
