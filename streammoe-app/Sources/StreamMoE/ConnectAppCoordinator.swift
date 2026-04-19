import SwiftUI
import AppKit
import StreamMoECore

// Owns the status-bar menu: header, model dropdown, Connect-an-app submenu,
// settings submenu (allow-LAN + Ollama toggle), and the welcome shortcut.
// Each pane is an NSWindow hosting a SwiftUI view; opening a pane calls
// retainListener() on the shared ServerStatus so polling only runs when
// something cares.
@MainActor
final class ConnectAppCoordinator {
    private weak var statusItem: NSStatusItem?
    private let serverStatus: ServerStatus
    private let detector = AppDetector()
    private let modelManager: ModelManager
    private let networkAccess: NetworkAccessSettings
    private let ollama: OllamaController
    private let supervisor: ServerSupervisor
    private let onShowWelcome: () -> Void
    private var openWindows: [String: NSWindow] = [:]
    private weak var currentMenu: NSMenu?

    init(statusItem: NSStatusItem,
         serverStatus: ServerStatus,
         modelManager: ModelManager,
         networkAccess: NetworkAccessSettings,
         ollama: OllamaController,
         supervisor: ServerSupervisor,
         onShowWelcome: @escaping () -> Void) {
        self.statusItem = statusItem
        self.serverStatus = serverStatus
        self.modelManager = modelManager
        self.networkAccess = networkAccess
        self.ollama = ollama
        self.supervisor = supervisor
        self.onShowWelcome = onShowWelcome
    }

    func installMenu() {
        let menu = NSMenu()
        menu.delegate = MenuRefresher.shared(rebuild: { [weak self] in self?.rebuildMenu() })
        menu.autoenablesItems = false
        currentMenu = menu
        rebuildInto(menu: menu)
        statusItem?.menu = menu
    }

    private func rebuildMenu() {
        guard let m = currentMenu else { return }
        m.removeAllItems()
        rebuildInto(menu: m)
    }

    private func rebuildInto(menu: NSMenu) {
        let online = serverStatus.isReachable
        let header = NSMenuItem(title: online
                                ? "StreamMoE · \(serverStatus.snapshot.id.isEmpty ? "(no model)" : serverStatus.snapshot.id)"
                                : "StreamMoE · offline", action: nil, keyEquivalent: "")
        header.isEnabled = false
        menu.addItem(header)
        // StreamMoE ON/OFF toggle. Enforces "exactly one of {StreamMoE,
        // Ollama} owns port 11434 at any given time."
        let toggleTitle: String
        let toggleSelector: Selector
        switch supervisor.state {
        case .running: toggleTitle = "Turn StreamMoE OFF"; toggleSelector = #selector(toggleServerOff)
        default:       toggleTitle = "Turn StreamMoE ON";  toggleSelector = #selector(toggleServerOn)
        }
        let toggleItem = NSMenuItem(title: toggleTitle, action: toggleSelector, keyEquivalent: "")
        toggleItem.target = self
        menu.addItem(toggleItem)

        menu.addItem(NSMenuItem.separator())

        // Model dropdown.
        let modelRoot = NSMenuItem(title: "Model", action: nil, keyEquivalent: "")
        let modelMenu = NSMenu()
        let local = modelManager.listLocal()
        if local.isEmpty {
            let none = NSMenuItem(title: "(no .gguf files in ~/StreamMoE/models)", action: nil, keyEquivalent: "")
            none.isEnabled = false
            modelMenu.addItem(none)
        } else {
            for spec in local {
                let item = NSMenuItem(title: spec.id, action: #selector(selectModel(_:)), keyEquivalent: "")
                item.target = self
                item.representedObject = spec.id
                item.state = (spec.id == serverStatus.snapshot.id) ? .on : .off
                modelMenu.addItem(item)
            }
        }
        modelRoot.submenu = modelMenu
        menu.addItem(modelRoot)

        // Connect-an-app submenu.
        let connectItem = NSMenuItem(title: "Connect an app", action: nil, keyEquivalent: "")
        let submenu = NSMenu()
        for pane in PaneKind.allCases {
            let item = NSMenuItem(title: pane.headline, action: #selector(openPane(_:)), keyEquivalent: "")
            item.target = self
            item.representedObject = pane.rawValue
            submenu.addItem(item)
        }
        connectItem.submenu = submenu
        menu.addItem(connectItem)

        // Settings submenu.
        let settingsItem = NSMenuItem(title: "Settings", action: nil, keyEquivalent: "")
        let settingsMenu = NSMenu()
        let lanToggle = NSMenuItem(
            title: "Allow network access",
            action: #selector(toggleAllowLAN(_:)),
            keyEquivalent: ""
        )
        lanToggle.target = self
        lanToggle.state = networkAccess.allowLAN ? .on : .off
        settingsMenu.addItem(lanToggle)

        let ollamaState = ollama.state()
        switch ollamaState {
        case .notInstalled:
            let it = NSMenuItem(title: "Ollama: not installed", action: nil, keyEquivalent: "")
            it.isEnabled = false
            settingsMenu.addItem(it)
        case .stopped:
            let it = NSMenuItem(title: "Start Ollama", action: #selector(startOllama), keyEquivalent: "")
            it.target = self
            settingsMenu.addItem(it)
        case .running:
            let it = NSMenuItem(title: "Stop Ollama (port 11434 conflict)",
                                action: #selector(stopOllama), keyEquivalent: "")
            it.target = self
            settingsMenu.addItem(it)
        }

        settingsMenu.addItem(NSMenuItem.separator())
        let welcome = NSMenuItem(title: "Show welcome…", action: #selector(showWelcome), keyEquivalent: "")
        welcome.target = self
        settingsMenu.addItem(welcome)
        settingsItem.submenu = settingsMenu
        menu.addItem(settingsItem)

        menu.addItem(NSMenuItem.separator())
        let quit = NSMenuItem(title: "Quit StreamMoE",
                              action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        menu.addItem(quit)
    }

    // Menu actions.
    @objc private func openPane(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String, let pane = PaneKind(rawValue: raw) else { return }
        showPane(pane)
        logPaneOpen(pane)
    }

    @objc private func selectModel(_ sender: NSMenuItem) {
        guard let id = sender.representedObject as? String,
              let spec = modelManager.listLocal().first(where: { $0.id == id }) else { return }
        // Restart the supervisor on the picked model; Ollama is already off
        // because the server takes port 11434.
        try? supervisor.restart(model: spec)
        rebuildMenu()
    }

    @objc private func toggleServerOn() {
        // (1) Free the port by stopping Ollama if it's running.
        if case .running = ollama.state() { try? ollama.stop() }
        // (2) Pick the first ready model with a sidecar, then the first
        // any-model, else bail with a warning — wizard handles onboarding.
        let models = modelManager.listLocal()
        guard let model = models.first(where: { $0.sidecarPresent }) ?? models.first else {
            let alert = NSAlert(); alert.messageText = "No model available"
            alert.informativeText = "Drop a .gguf into ~/StreamMoE/models, then try again."
            alert.runModal(); return
        }
        try? supervisor.start(model: model)
        rebuildMenu()
    }

    @objc private func toggleServerOff() {
        try? supervisor.stop()
        rebuildMenu()
    }

    @objc private func toggleAllowLAN(_ sender: NSMenuItem) {
        networkAccess.allowLAN.toggle()
        rebuildMenu()
    }

    @objc private func startOllama() {
        try? ollama.start(); rebuildMenu()
    }

    @objc private func stopOllama() {
        try? ollama.stop(); rebuildMenu()
    }

    @objc private func showWelcome() { onShowWelcome() }

    // Pane window management.
    private func showPane(_ pane: PaneKind) {
        if let existing = openWindows[pane.rawValue] {
            existing.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true); return
        }
        let detected = detector.snapshot()
        let content = PaneRoot(pane: pane, detected: detected)
            .environmentObject(serverStatus)
        let hosting = NSHostingController(rootView: content)
        let window = NSWindow(contentViewController: hosting)
        window.title = pane.headline
        window.styleMask = [.titled, .closable, .miniaturizable]
        window.setContentSize(NSSize(width: 520, height: 460))
        window.center()
        window.isReleasedWhenClosed = false

        serverStatus.retainListener()
        let close = NotificationCenter.default.addObserver(
            forName: NSWindow.willCloseNotification, object: window, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.serverStatus.releaseListener()
                self?.openWindows.removeValue(forKey: pane.rawValue)
            }
        }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        openWindows[pane.rawValue] = window
        objc_setAssociatedObject(window, "closeObserver", close, .OBJC_ASSOCIATION_RETAIN)
    }

    private func logPaneOpen(_ pane: PaneKind) {
        let url = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Library/Logs/StreamMoE/pane-opens.log")
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        let line = "\(ISO8601DateFormatter().string(from: Date())) pane=\(pane.rawValue)\n"
        if let data = line.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: url.path) {
                if let handle = try? FileHandle(forWritingTo: url) {
                    try? handle.seekToEnd(); try? handle.write(contentsOf: data); try? handle.close()
                }
            } else { try? data.write(to: url) }
        }
    }
}

// NSMenu delegate that fires a rebuild right before the menu opens, so
// reachability + model list + ollama state stay fresh without polling.
final class MenuRefresher: NSObject, NSMenuDelegate {
    private static var stored: MenuRefresher?
    private let rebuild: () -> Void
    init(rebuild: @escaping () -> Void) { self.rebuild = rebuild }
    static func shared(rebuild: @escaping () -> Void) -> MenuRefresher {
        if let s = stored { return s }
        let s = MenuRefresher(rebuild: rebuild)
        stored = s
        return s
    }
    func menuWillOpen(_ menu: NSMenu) { rebuild() }
}

enum PaneKind: String, CaseIterable {
    case openWebUI, cursor, lmStudio, claudeDesktop, other
    var headline: String {
        switch self {
        case .openWebUI:     return "Open WebUI"
        case .cursor:        return "Cursor"
        case .lmStudio:      return "LM Studio"
        case .claudeDesktop: return "Claude Desktop"
        case .other:         return "Other app"
        }
    }
    var guideResource: String {
        switch self {
        case .openWebUI:     return "OpenWebUI"
        case .cursor:        return "Cursor"
        case .lmStudio:      return "LMStudio"
        case .claudeDesktop: return "ClaudeDesktop"
        case .other:         return "Other"
        }
    }
}
