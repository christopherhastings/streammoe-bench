import SwiftUI
import AppKit
import StreamMoECore

// Menu-bar-only SwiftUI app. No dock icon, no window at launch —
// the UI surface is the NSMenu owned by ConnectAppCoordinator plus
// an on-demand Welcome wizard window on first run.
@main
struct StreamMoEApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        Settings { EmptyView() }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem!
    private var coordinator: ConnectAppCoordinator!
    private var serverStatus: ServerStatus!
    private var modelManager: ModelManager!
    private var networkAccess: NetworkAccessSettings!
    private var ollama: OllamaController!
    private var supervisor: ServerSupervisor!
    private var welcomeWindow: NSWindow?

    nonisolated func applicationDidFinishLaunching(_: Notification) {
        MainActor.assumeIsolated {
            NSApp.setActivationPolicy(.accessory)
            self.serverStatus = ServerStatus()
            self.modelManager = ModelManager(
                modelsDir: URL(fileURLWithPath: NSHomeDirectory() + "/StreamMoE/models")
            )
            self.networkAccess = NetworkAccessSettings()
            self.ollama = OllamaController()
            // Path resolution: prefer a bundled llama-server next to the .app;
            // fall back to the dev build path. Users installing from GitHub
            // get the bundled one; dev runs of `swift run StreamMoE` use the
            // fork build tree.
            let bundledServer = Bundle.main.bundlePath + "/Contents/MacOS/llama-server"
            let devServer = "/Users/claude/streammoe/anemll-flash-llama.cpp/build/bin/llama-server"
            let serverBin = FileManager.default.fileExists(atPath: bundledServer) ? bundledServer : devServer
            self.supervisor = ServerSupervisor(binary: serverBin, network: self.networkAccess)
            self.statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
            self.statusItem.button?.title = "StreamMoE"
            self.coordinator = ConnectAppCoordinator(
                statusItem: self.statusItem,
                serverStatus: self.serverStatus,
                modelManager: self.modelManager,
                networkAccess: self.networkAccess,
                ollama: self.ollama,
                supervisor: self.supervisor,
                onShowWelcome: { [weak self] in self?.showWelcome() }
            )
            self.coordinator.installMenu()
            self.serverStatus.startPolling()

            // First-run: show welcome if no models are currently known to
            // /streammoe/status AND the user hasn't been shown it before.
            if !UserDefaults.standard.bool(forKey: "com.streammoe.welcome.shown") {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                    self?.showWelcome()
                }
            }
        }
    }

    private func showWelcome() {
        if let existing = welcomeWindow {
            existing.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true); return
        }
        let bridge = WizardBridge()
        let view = WelcomeWizardView(
            wizard: bridge,
            models: modelManager.listLocal(),
            onPrepare: { _ in
                // Placeholder: ModelDownloader is a later concern. Simulate
                // a 1-second warmup for the UI demo, then flip to done.
                Task { @MainActor [weak bridge] in
                    for step in stride(from: 0.0, through: 1.0, by: 0.1) {
                        bridge?.reportProgress(step)
                        try? await Task.sleep(nanoseconds: 100_000_000)
                    }
                    bridge?.completePreparation()
                }
            },
            onFinish: { [weak self] in
                UserDefaults.standard.set(true, forKey: "com.streammoe.welcome.shown")
                self?.welcomeWindow?.close()
                self?.welcomeWindow = nil
            }
        )
        let hosting = NSHostingController(rootView: view)
        let w = NSWindow(contentViewController: hosting)
        w.title = "StreamMoE Welcome"
        w.styleMask = [.titled, .closable]
        w.isReleasedWhenClosed = false
        w.center()
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        welcomeWindow = w
    }
}
