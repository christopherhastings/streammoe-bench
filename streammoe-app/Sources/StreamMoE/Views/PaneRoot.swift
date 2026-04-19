import SwiftUI

// Top-level pane view. Picks the per-app subview and renders the detailed-guide
// disclosure + feedback footer that are common to all five panes.
struct PaneRoot: View {
    let pane: PaneKind
    let detected: AppDetector.Cache
    @EnvironmentObject private var status: ServerStatus
    @State private var showGuide = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Connection indicator.
            HStack(spacing: 8) {
                Circle()
                    .fill(status.isReachable ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                Text(status.isReachable
                     ? "StreamMoE server: \(status.snapshot.listening_address)"
                     : "StreamMoE server not reachable on 11434")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 18)
            .padding(.top, 14)

            Divider().padding(.top, 10)

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Group {
                        switch pane {
                        case .openWebUI:     OpenWebUIPane(detected: detected)
                        case .cursor:        CursorPane(detected: detected)
                        case .lmStudio:      LMStudioPane(detected: detected)
                        case .claudeDesktop: ClaudeDesktopPane(detected: detected)
                        case .other:         OtherPane(detected: detected)
                        }
                    }
                    .environmentObject(status)
                }
                .padding(18)
            }

            Divider()

            HStack {
                Button(action: { showGuide.toggle() }) {
                    Label(showGuide ? "Hide detailed guide" : "Detailed guide", systemImage: "book")
                }
                .buttonStyle(.link)
                Spacer()
                Button("I still need help", action: openFeedback)
                    .buttonStyle(.link)
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 10)

            if showGuide {
                Divider()
                GuideView(resource: pane.guideResource)
                    .frame(maxHeight: 240)
            }
        }
        .frame(minWidth: 480)
    }

    private func openFeedback() {
        // Prefill a GitHub Issue rather than running our own intake server.
        let title = "\(pane.headline): connection help".addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        let body = """
        # App: \(pane.headline)
        # StreamMoE server reachable: \(status.isReachable ? "yes" : "no")
        # Model id: \(status.snapshot.id)
        # Model path: \(status.snapshot.model_path)

        ## What I tried

        ## What happened
        """.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        let url = URL(string: "https://github.com/streammoe/streammoe/issues/new?title=\(title)&body=\(body)")!
        NSWorkspace.shared.open(url)
    }
}
