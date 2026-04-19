import XCTest
@testable import StreamMoECore

final class NetworkAccessSettingsTests: XCTestCase {
    private func makeSUT() -> (NetworkAccessSettings, UserDefaults) {
        // Each test gets a fresh ephemeral UserDefaults suite — avoids
        // persistent state bleed between runs and on dev machines.
        let suite = "test-streammoe-" + UUID().uuidString
        let d = UserDefaults(suiteName: suite)!
        return (NetworkAccessSettings(defaults: d), d)
    }

    func testAllowLANDefaultsFalse() {
        let (sut, _) = makeSUT()
        XCTAssertFalse(sut.allowLAN)
    }

    func testListenHostIsLoopbackWhenLANDisabled() {
        let (sut, _) = makeSUT()
        sut.allowLAN = false
        XCTAssertEqual(sut.listenHost, "127.0.0.1")
    }

    func testListenHostIsWildcardWhenLANEnabled() {
        let (sut, _) = makeSUT()
        sut.allowLAN = true
        XCTAssertEqual(sut.listenHost, "0.0.0.0")
    }

    func testPersistsAcrossInstances() {
        let (sut, defaults) = makeSUT()
        sut.allowLAN = true
        let reloaded = NetworkAccessSettings(defaults: defaults)
        XCTAssertTrue(reloaded.allowLAN)
    }
}
