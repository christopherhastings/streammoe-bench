import Foundation

// Backs the "Allow network access" toggle referenced in the Open WebUI copy.
// Stored in UserDefaults so it survives across launches without a custom
// settings file. When enabled, llama-server is spawned with --host 0.0.0.0
// so other machines on the LAN can reach it; default is loopback-only.
public final class NetworkAccessSettings: @unchecked Sendable {
    private let defaults: UserDefaults
    private let key = "com.streammoe.network.allowLAN"

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    public var allowLAN: Bool {
        get { defaults.bool(forKey: key) }
        set { defaults.set(newValue, forKey: key) }
    }

    public var listenHost: String {
        allowLAN ? "0.0.0.0" : "127.0.0.1"
    }
}
