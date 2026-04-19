import Foundation

public enum OpenWebUIStatus: Equatable, Sendable {
    case notDetected
    case detectedNoStreammoe(base: String)
    case detectedWithStreammoe(base: String, modelID: String)
}

// Probes local Open WebUI installs to decide what copy to surface on the
// Connect-an-app → Open WebUI pane. The pane otherwise has to assume the
// user hasn't set anything up — this lets us show the green-check path
// when they already have.
//
// Detection order: the UI is usually at 3000 (Docker) or 8080 (native).
// For each base, we (1) GET a lightweight admin endpoint to confirm it
// really is Open WebUI and not some other service, then (2) ask for its
// configured Ollama base URLs and look for anything ending in :11434.
public struct OpenWebUIProbe: Sendable {
    public static let candidateBases = ["http://localhost:3000", "http://localhost:8080"]

    private let client: HTTPClient
    private let bases: [String]

    public init(client: HTTPClient = URLSessionHTTPClient(),
                bases: [String] = OpenWebUIProbe.candidateBases) {
        self.client = client
        self.bases = bases
    }

    public func status(modelID: String) async -> OpenWebUIStatus {
        for base in bases {
            if let detected = await probe(base: base, modelID: modelID) {
                return detected
            }
        }
        return .notDetected
    }

    private func probe(base: String, modelID: String) async -> OpenWebUIStatus? {
        guard let adminURL = URL(string: base + "/api/v1/auths/admin/details") else { return nil }
        // If this returns anything at all (even 401), something OWUI-shaped
        // is listening. Pure connection refusal → move on.
        guard (try? await client.get(adminURL, timeout: 1.0)) != nil else {
            return nil
        }

        var connected = false
        let urlEndpoints = [base + "/ollama/urls", base + "/api/v1/openai/urls"]
        for str in urlEndpoints {
            guard let u = URL(string: str) else { continue }
            guard let (data, status) = try? await client.get(u, timeout: 1.0), status == 200 else { continue }
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                for key in ["OLLAMA_BASE_URLS", "OPENAI_API_BASE_URLS"] {
                    if let arr = json[key] as? [String],
                       arr.contains(where: { $0.contains(":11434") }) {
                        connected = true
                    }
                }
            }
        }
        return connected ? .detectedWithStreammoe(base: base, modelID: modelID)
                         : .detectedNoStreammoe(base: base)
    }
}
