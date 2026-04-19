import SwiftUI

struct ClaudeDesktopPane: View {
    let detected: AppDetector.Cache

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Claude Desktop")
                .font(.title2).bold()

            if !detected.claudeDesktopInstalled {
                Text("Claude Desktop doesn't appear to be installed in /Applications. This pane is still informational.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text("Claude Desktop uses Anthropic's hosted API — it doesn't support local models directly.")
                .font(.body)

            Text("To use StreamMoE with Claude Desktop, you'd need an MCP bridge. This isn't a one-click setup. See the guide below if you want to try it.")
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
