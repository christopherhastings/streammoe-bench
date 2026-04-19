import SwiftUI

struct CursorPane: View {
    let detected: AppDetector.Cache
    @EnvironmentObject private var status: ServerStatus

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Cursor")
                .font(.title2).bold()

            if !detected.cursorInstalled {
                Text("Cursor doesn't appear to be installed in /Applications. Install from cursor.sh, then return here.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            Text("Cursor uses OpenAI-compatible endpoints. Add StreamMoE as a custom model.")
                .font(.body)

            VStack(alignment: .leading, spacing: 8) {
                CopyButton(label: "Settings path", value: "Cursor → Settings → Models → Add Model → OpenAI API Key")
                CopyButton(label: "Base URL",      value: "http://localhost:11434/v1")
                CopyButton(label: "API Key",       value: "sk-streammoe")
                CopyButton(label: "Model name",    value: status.snapshot.id)
            }
            .padding(.top, 4)

            Text("Cursor requires the model name to match exactly. Use Copy — don't type.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
