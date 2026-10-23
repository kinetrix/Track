import AppKit
import SwiftUI

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct FaceHandTrackApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)  // console lines reach a log file at once, not when the buffer fills
        // Command-line checks (see SelfTest.swift) run without opening a window.
        if let code = SelfTest.runIfRequested() { exit(code) }
    }

    var body: some Scene {
        WindowGroup("FaceHandTrack") {
            ContentView()
        }
        .defaultSize(width: 1280, height: 800)
    }
}
