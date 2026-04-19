import SwiftUI

struct LMStudioPane: View {
    let detected: AppDetector.Cache
    @EnvironmentObject private var status: ServerStatus

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("LM Studio")
                .font(.title2).bold()

            if !detected.lmStudioInstalled {
                Text("LM Studio doesn't appear to be installed in /Applications. Install from lmstudio.ai, then return here.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            Text("Add StreamMoE as a remote OpenAI endpoint in LM Studio.")
                .font(.body)

            VStack(alignment: .leading, spacing: 8) {
                CopyButton(label: "Settings path", value: "Chat tab → model dropdown → Add Custom Endpoint")
                CopyButton(label: "Base URL",      value: "http://localhost:11434/v1")
                CopyButton(label: "API Key",       value: "sk-streammoe")
                CopyButton(label: "Model name",    value: status.snapshot.id)
            }
            .padding(.top, 4)

            Text("LM Studio's own server runs on port 1234. StreamMoE is on 11434. Both can run simultaneously.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
