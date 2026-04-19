import SwiftUI
import StreamMoECore

// Presents each step of the Welcome wizard. Behavior-free — only renders
// the current state and emits user intents back to a WelcomeWizardState
// via the bridging @ObservedObject wrapper below. The state machine
// itself has XCTest coverage in StreamMoECoreTests.
struct WelcomeWizardView: View {
    @ObservedObject var wizard: WizardBridge
    let models: [ModelSpec]
    let onPrepare: (ModelSpec) -> Void
    let onFinish: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Welcome to StreamMoE")
                .font(.largeTitle).bold()
            switch wizard.step {
            case .intro:
                introBody
            case .pickModel:
                pickBody
            case .preparing(let model, let progress):
                prepareBody(model: model, progress: progress)
            case .done(let model):
                doneBody(model: model)
            }
            Spacer()
        }
        .padding(28)
        .frame(width: 540, height: 380)
    }

    private var introBody: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("StreamMoE replaces Ollama on port 11434 with a purpose-built server that streams MoE experts from SSD. Your existing Ollama-compatible apps keep working.")
                .font(.body)
            Text("One-time setup: pick a model and click Prepare.")
                .foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Let's go") { wizard.state.advance() }
                    .keyboardShortcut(.defaultAction)
            }
        }
    }

    private var pickBody: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Pick a model").font(.title3).bold()
            if models.isEmpty {
                Text("No .gguf models found in your models directory. Drop a GGUF into ~/StreamMoE/models and re-open StreamMoE.")
                    .foregroundStyle(.secondary)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(models, id: \.id) { m in
                            Button {
                                wizard.state.pick(m)
                                onPrepare(m)
                            } label: {
                                HStack {
                                    VStack(alignment: .leading) {
                                        Text(m.id).font(.body.monospaced())
                                        Text(formatSize(m.sizeBytes) +
                                             (m.sidecarPresent ? " · sidecar ready" : " · sidecar missing"))
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    Image(systemName: "chevron.right")
                                }
                                .padding(10)
                                .background(RoundedRectangle(cornerRadius: 8)
                                                .fill(Color(nsColor: .controlBackgroundColor)))
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
        }
    }

    private func prepareBody(model: ModelSpec, progress: Double) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Preparing \(model.id)").font(.title3).bold()
            ProgressView(value: progress)
            Text(progress >= 1.0
                 ? "Warming up…"
                 : "Faulting in the sidecar and warming kernels. First time only.")
                .foregroundStyle(.secondary)
        }
    }

    private func doneBody(model: ModelSpec) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("You're set").font(.title3).bold()
            Text("StreamMoE is listening on port 11434. Open your favorite Ollama-compatible app and point it at this Mac.")
            HStack {
                Spacer()
                Button("Done", action: onFinish).keyboardShortcut(.defaultAction)
            }
        }
    }

    private func formatSize(_ bytes: Int64) -> String {
        let gib = Double(bytes) / 1024 / 1024 / 1024
        return String(format: "%.1f GiB", gib)
    }
}

// SwiftUI needs an observable wrapper; WelcomeWizardState is in Core and
// deliberately Combine-free to keep tests fast. This bridge publishes step
// changes for SwiftUI redraws.
@MainActor
final class WizardBridge: ObservableObject {
    let state: WelcomeWizardState
    @Published private(set) var step: WelcomeStep

    init(state: WelcomeWizardState = WelcomeWizardState()) {
        self.state = state
        self.step = state.step
    }

    func sync() { self.step = state.step }

    func advance()    { state.advance();          sync() }
    func pick(_ m: ModelSpec) { state.pick(m);    sync() }
    func reportProgress(_ p: Double) { state.reportProgress(p); sync() }
    func completePreparation() { state.completePreparation(); sync() }
    func reset() { state.reset(); sync() }
}
