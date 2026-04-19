import SwiftUI

struct OtherPane: View {
    let detected: AppDetector.Cache
    @EnvironmentObject private var status: ServerStatus

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Other app")
                .font(.title2).bold()

            Text("StreamMoE speaks two protocols. Use whichever your app expects.")
                .font(.body)

            VStack(alignment: .leading, spacing: 8) {
                CopyButton(label: "OpenAI-compat", value: "http://localhost:11434/v1")
                CopyButton(label: "Ollama-native", value: "http://localhost:11434")
                CopyButton(label: "Model",         value: status.snapshot.id)
                CopyButton(label: "API key",       value: "any non-empty string")
            }
            .padding(.top, 4)

            Text("If your app hardcoded Ollama's endpoints, it already works — we're on the same port with the same API.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
