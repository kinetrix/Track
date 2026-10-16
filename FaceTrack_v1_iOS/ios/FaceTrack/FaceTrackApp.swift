import SwiftUI

@main
struct FaceTrackApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .statusBarHidden()
                .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
        }
    }
}
