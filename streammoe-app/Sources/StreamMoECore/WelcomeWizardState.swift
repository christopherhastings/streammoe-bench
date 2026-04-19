import Foundation

public enum WelcomeStep: Equatable, Sendable {
    case intro
    case pickModel
    case preparing(model: ModelSpec, progress: Double)
    case done(model: ModelSpec)
}

// State machine for the first-run wizard. UI-agnostic — SwiftUI view observes
// `step` and renders the matching screen. Tests drive the transitions directly
// without having to mount any view hierarchy.
public final class WelcomeWizardState: @unchecked Sendable {
    private(set) public var step: WelcomeStep = .intro

    public init() {}

    public func advance() {
        switch step {
        case .intro: step = .pickModel
        default: break  // linear flow; further advance() calls are no-ops
        }
    }

    public func pick(_ model: ModelSpec) {
        guard case .pickModel = step else { return }
        step = .preparing(model: model, progress: 0)
    }

    public func reportProgress(_ p: Double) {
        guard case .preparing(let m, _) = step else { return }
        step = .preparing(model: m, progress: max(0, min(1, p)))
    }

    public func completePreparation() {
        guard case .preparing(let m, _) = step else { return }
        step = .done(model: m)
    }

    public func reset() { step = .intro }
}
