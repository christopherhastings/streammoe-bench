import Foundation

// Minimal seam. Injected into probes so tests can canned-respond per URL
// without booting a real HTTP server. The production implementation wraps
// URLSession. Returns (body, statusCode).
public protocol HTTPClient: Sendable {
    func get(_ url: URL, timeout: TimeInterval) async throws -> (Data, Int)
}

public struct URLSessionHTTPClient: HTTPClient {
    public init() {}
    public func get(_ url: URL, timeout: TimeInterval = 2.0) async throws -> (Data, Int) {
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.timeoutInterval = timeout
        let (data, resp) = try await URLSession.shared.data(for: req)
        return (data, (resp as? HTTPURLResponse)?.statusCode ?? -1)
    }
}
