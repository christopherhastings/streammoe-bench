import Foundation

// Plain value type returned by /streammoe/status. Kept in Core so tests can
// construct it without touching any Combine / ObservableObject plumbing.
public struct ServerStatusSnapshot: Codable, Equatable, Sendable {
    public var ok: Bool
    public var id: String
    public var model_path: String
    public var moe_sidecar_path: String
    public var moe_mode: String
    public var moe_eager_load: Bool
    public var streammoe_warmup: Bool
    public var moe_keep_warm: Bool
    public var moe_keep_warm_interval_s: Int
    public var mlock: Bool
    public var n_batch: Int
    public var n_ubatch: Int
    public var n_ctx: Int
    public var n_gpu_layers: Int
    public var server_uptime_s: Int
    public var listening_address: String

    public init(ok: Bool = false, id: String = "", model_path: String = "",
                moe_sidecar_path: String = "", moe_mode: String = "",
                moe_eager_load: Bool = false, streammoe_warmup: Bool = false,
                moe_keep_warm: Bool = false, moe_keep_warm_interval_s: Int = 0,
                mlock: Bool = false, n_batch: Int = 0, n_ubatch: Int = 0,
                n_ctx: Int = 0, n_gpu_layers: Int = 0,
                server_uptime_s: Int = 0, listening_address: String = "") {
        self.ok = ok
        self.id = id
        self.model_path = model_path
        self.moe_sidecar_path = moe_sidecar_path
        self.moe_mode = moe_mode
        self.moe_eager_load = moe_eager_load
        self.streammoe_warmup = streammoe_warmup
        self.moe_keep_warm = moe_keep_warm
        self.moe_keep_warm_interval_s = moe_keep_warm_interval_s
        self.mlock = mlock
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.server_uptime_s = server_uptime_s
        self.listening_address = listening_address
    }
}
