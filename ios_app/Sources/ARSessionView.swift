import ARKit
import SwiftUI

/// SwiftUI wrapper around `ARSCNView`.
///
/// `ARSCNView` creates and owns its own `ARSession` internally — replacing
/// `view.session` with a foreign session breaks the view's rendering pipeline
/// and produces a black screen with `FigCaptureSourceRemote` failures. So we
/// hand the view's own session to the recorder via `attach(to:)` instead.
struct ARSessionView: UIViewRepresentable {
    let recorder: CaptureRecorder

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.automaticallyUpdatesLighting = true
        view.preferredFramesPerSecond = 60
        view.backgroundColor = .black

        // Defer attach to the next runloop tick. Calling it inline here would
        // mutate `recorder`'s @Published properties (phase/status) during the
        // current SwiftUI view-update pass, which logs
        //   "Publishing changes from within view updates is not allowed".
        let recorder = self.recorder
        DispatchQueue.main.async {
            recorder.attach(to: view.session)
        }
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
