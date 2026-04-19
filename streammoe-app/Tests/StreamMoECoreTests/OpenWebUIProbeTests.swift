import XCTest
@testable import StreamMoECore

final class FakeHTTPClient: HTTPClient, @unchecked Sendable {
    struct Response { let status: Int; let body: Data }
    var responses: [URL: Response] = [:]
    var errors: [URL: Error] = [:]
    private(set) var requested: [URL] = []

    func get(_ url: URL, timeout: TimeInterval) async throws -> (Data, Int) {
        requested.append(url)
        if let err = errors[url] { throw err }
        if let r = responses[url] { return (r.body, r.status) }
        throw URLError(.cannotConnectToHost)
    }
}

final class OpenWebUIProbeTests: XCTestCase {
    private func okJSON(_ s: String) -> FakeHTTPClient.Response {
        .init(status: 200, body: s.data(using: .utf8)!)
    }

    func testStatusNotDetectedWhenRootGETFails() async {
        let http = FakeHTTPClient()
        // Both the docker-internal and localhost probes fail.
        http.errors[URL(string: "http://localhost:3000/api/v1/auths/admin/details")!] = URLError(.cannotConnectToHost)
        http.errors[URL(string: "http://localhost:8080/api/v1/auths/admin/details")!] = URLError(.cannotConnectToHost)
        let sut = OpenWebUIProbe(client: http)
        let result = await sut.status(modelID: "qwen-35b")
        XCTAssertEqual(result, .notDetected)
    }

    func testDetectedNoStreamMoEWhenAdminReachableButNoOllamaConn() async {
        let http = FakeHTTPClient()
        http.responses[URL(string: "http://localhost:3000/api/v1/auths/admin/details")!] = okJSON("{}")
        // Ollama endpoints list returns no 11434 entry.
        http.responses[URL(string: "http://localhost:3000/api/v1/openai/urls")!] =
            okJSON(#"{"OPENAI_API_BASE_URLS":["https://api.openai.com/v1"]}"#)
        http.responses[URL(string: "http://localhost:3000/ollama/urls")!] =
            okJSON(#"{"OLLAMA_BASE_URLS":[]}"#)
        let sut = OpenWebUIProbe(client: http)
        let result = await sut.status(modelID: "qwen-35b")
        XCTAssertEqual(result, .detectedNoStreammoe(base: "http://localhost:3000"))
    }

    func testDetectedWithStreamMoEWhenOllamaURLIs11434() async {
        let http = FakeHTTPClient()
        http.responses[URL(string: "http://localhost:3000/api/v1/auths/admin/details")!] = okJSON("{}")
        http.responses[URL(string: "http://localhost:3000/api/v1/openai/urls")!] =
            okJSON(#"{"OPENAI_API_BASE_URLS":[]}"#)
        http.responses[URL(string: "http://localhost:3000/ollama/urls")!] =
            okJSON(#"{"OLLAMA_BASE_URLS":["http://host.docker.internal:11434"]}"#)
        let sut = OpenWebUIProbe(client: http)
        let result = await sut.status(modelID: "qwen-35b")
        XCTAssertEqual(result, .detectedWithStreammoe(base: "http://localhost:3000", modelID: "qwen-35b"))
    }
}
