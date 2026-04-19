import SwiftUI
import StreamMoECore

struct OpenWebUIPane: View {
    let detected: AppDetector.Cache
    @EnvironmentObject private var status: ServerStatus
    @State private var probeStatus: OpenWebUIStatus = .notDetected

    private var preferredURL: String {
        detected.openWebUIInDocker ? "http://host.docker.internal:11434" : "http://localhost:11434"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Open WebUI").font(.title2).bold()

            // Live green-check if Open WebUI already points at StreamMoE.
            switch probeStatus {
            case .detectedWithStreammoe(let base, let modelID):
                Label("Already connected — refresh the model dropdown and pick \(modelID).",
                      systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("Found Open WebUI at \(base)").font(.caption).foregroundStyle(.secondary)
            case .detectedNoStreammoe(let base):
                Text("Open WebUI detected at \(base). Add a StreamMoE connection in Admin Panel → Settings → Connections.")
            case .notDetected:
                Text("Add a connection in Admin Panel → Settings → Connections.")
            }

            VStack(alignment: .leading, spacing: 8) {
                CopyButton(label: "Type",  value: "Ollama API")
                CopyButton(label: "URL",   value: preferredURL)
                CopyButton(label: "Model", value: status.snapshot.id)
            }
            .padding(.top, 4)

            Text("If Open WebUI is on another computer, use this Mac's LAN IP and enable 'Allow network access' in StreamMoE Settings.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .task(id: status.snapshot.id) {
            // Re-probe whenever the bound model id changes so the green-check
            // tracks the actual advertised id, not a stale snapshot.
            let probe = OpenWebUIProbe()
            probeStatus = await probe.status(modelID: status.snapshot.id)
        }
    }
}
