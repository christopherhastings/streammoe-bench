import XCTest
@testable import StreamMoECore

final class WelcomeWizardStateTests: XCTestCase {
    func testStartsAtIntro() {
        let sut = WelcomeWizardState()
        XCTAssertEqual(sut.step, .intro)
    }

    func testAdvanceFromIntroGoesToPickModel() {
        let sut = WelcomeWizardState()
        sut.advance()
        XCTAssertEqual(sut.step, .pickModel)
    }

    func testPickingAModelMovesToPreparing() {
        let sut = WelcomeWizardState()
        sut.advance()  // pickModel
        let spec = ModelSpec(id: "qwen-35b", path: "/tmp/x.gguf",
                             sizeBytes: 1, sidecarPresent: true, sidecarPath: "/tmp/x-sidecar")
        sut.pick(spec)
        XCTAssertEqual(sut.step, .preparing(model: spec, progress: 0))
    }

    func testReportingProgressUpdatesValue() {
        let sut = WelcomeWizardState()
        sut.advance()
        let spec = ModelSpec(id: "qwen-35b", path: "/tmp/x.gguf",
                             sizeBytes: 1, sidecarPresent: true, sidecarPath: "/tmp/x-sidecar")
        sut.pick(spec)
        sut.reportProgress(0.5)
        XCTAssertEqual(sut.step, .preparing(model: spec, progress: 0.5))
    }

    func testCompletingPreparationGoesToDone() {
        let sut = WelcomeWizardState()
        sut.advance()
        let spec = ModelSpec(id: "qwen-35b", path: "/tmp/x.gguf",
                             sizeBytes: 1, sidecarPresent: true, sidecarPath: "/tmp/x-sidecar")
        sut.pick(spec)
        sut.completePreparation()
        XCTAssertEqual(sut.step, .done(model: spec))
    }

    func testResetReturnsToIntro() {
        let sut = WelcomeWizardState()
        sut.advance(); sut.advance()
        sut.reset()
        XCTAssertEqual(sut.step, .intro)
    }

    // Invalid transitions are no-ops; the wizard's flow is linear enough
    // that we'd rather swallow a misordered event than crash the UI.
    func testPickingWhileAtIntroIsIgnored() {
        let sut = WelcomeWizardState()
        let spec = ModelSpec(id: "qwen-35b", path: "/tmp/x.gguf",
                             sizeBytes: 1, sidecarPresent: true, sidecarPath: nil)
        sut.pick(spec)
        XCTAssertEqual(sut.step, .intro)
    }
}
