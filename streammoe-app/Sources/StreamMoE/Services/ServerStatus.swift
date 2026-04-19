import Foundation
import Combine
import StreamMoECore

// Observable snapshot of /streammoe/status. Drives the copy-button values in
// each connect-an-app pane. Polls every 2s while at least one pane is open.
// Snapshot type lives in StreamMoECore so tests and headless code can
// construct it without pulling in Combine / AppKit.
@MainActor
final class ServerStatus: ObservableObject {
    typealias Snapshot = ServerStatusSnapshot

    @Published private(set) var snapshot = Snapshot()
    @Published private(set) var isReachable = false
    @Published private(set) var lastErrorDescription: String? = nil

    private var timer: Timer? = nil
    private var listenerCount = 0
    private let endpoint = URL(string: "http://127.0.0.1:11434/streammoe/status")!

    func retainListener() {
        listenerCount += 1
        if listenerCount == 1 { startPolling() }
    }

    func releaseListener() {
        listenerCount = max(0, listenerCount - 1)
        if listenerCount == 0 { stopPolling() }
    }

    func startPolling() {
        stopPolling()
        refresh()
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.refresh() }
        }
    }

    func stopPolling() {
        timer?.invalidate()
        timer = nil
    }

    func refresh() {
        var req = URLRequest(url: endpoint, timeoutInterval: 1.0)
        req.httpMethod = "GET"
        URLSession.shared.dataTask(with: req) { [weak self] data, resp, err in
            Task { @MainActor in
                guard let self else { return }
                if let err {
                    self.isReachable = false
                    self.lastErrorDescription = err.localizedDescription
                    return
                }
                guard let data, let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                    self.isReachable = false
                    self.lastErrorDescription = "non-200 from status endpoint"
                    return
                }
                do {
                    let snap = try JSONDecoder().decode(Snapshot.self, from: data)
                    self.snapshot = snap
                    self.isReachable = true
                    self.lastErrorDescription = nil
                } catch {
                    self.isReachable = false
                    self.lastErrorDescription = "decode: \(error.localizedDescription)"
                }
            }
        }.resume()
    }
}
