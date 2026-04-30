import ARKit
import SwiftUI
import WebKit

struct ContentView: View {
    @StateObject private var log = LogStream()
    @StateObject private var recorder: CaptureRecorder
    @StateObject private var uploader: Uploader

    @State private var showWebView = false
    @State private var webURL: URL? = nil
    @State private var zippedBundle: URL? = nil
    @State private var zipBytes: Int64 = 0
    @State private var showLogs: Bool = true

    init() {
        let log = LogStream()
        _log = StateObject(wrappedValue: log)
        _recorder = StateObject(wrappedValue: CaptureRecorder(log: log))
        _uploader = StateObject(wrappedValue: Uploader(log: log))
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            // Always render the AR view so `attach(to:)` runs on first
            // appearance — that's what kicks off the AR session. Hiding it
            // conditionally would mean the session never starts. When the
            // user toggles the camera off we just dim it and overlay text.
            ARSessionView(recorder: recorder)
                .ignoresSafeArea()
                .opacity(recorder.cameraOn ? 1 : 0.0)
            if !recorder.cameraOn {
                Color(white: 0.08).ignoresSafeArea()
                Text("Camera off")
                    .font(.title3.bold())
                    .foregroundColor(.white.opacity(0.5))
            }
            VStack(spacing: 0) {
                topBar
                Spacer()
                if showLogs {
                    LogPanel(log: log)
                        .frame(maxHeight: 220)
                        .padding(.horizontal, 12)
                }
                bottomPanel
            }
        }
        .onDisappear { recorder.pause() }
        .sheet(isPresented: $showWebView) {
            if let url = webURL {
                ViewerWebView(url: url, log: log)
                    .ignoresSafeArea()
            }
        }
    }

    // MARK: Top bar

    private var topBar: some View {
        HStack(spacing: 8) {
            tag(text: recorder.lidarSupported ? "TIER 1 · LiDAR" : "TIER 2 · VIO",
                color: recorder.lidarSupported ? .green : .orange)
            // Tracking quality. Until this turns green, ARKit's LiDAR fusion
            // refuses to integrate frames so REC will produce an empty mesh.
            tag(text: recorder.trackingState,
                color: recorder.trackingOK ? .green : .yellow)
            if case .recording = recorder.phase {
                tag(text: "REC \(recorder.frameCount) · \(formatBytes(recorder.sessionBytes))",
                    color: .red, dot: true)
            } else if recorder.frameCount > 0 {
                tag(text: "\(recorder.frameCount) frames · \(formatBytes(recorder.sessionBytes))",
                    color: .gray)
            }
            Spacer()
            Button(action: { recorder.toggleCamera() }) {
                Image(systemName: recorder.cameraOn ? "video.fill" : "video.slash.fill")
                    .frame(width: 36, height: 36)
                    .background(Color.black.opacity(0.55))
                    .foregroundColor(.white)
                    .clipShape(Circle())
            }
            Button(action: { showLogs.toggle() }) {
                Image(systemName: showLogs ? "text.justify" : "text.justify.left")
                    .frame(width: 36, height: 36)
                    .background(Color.black.opacity(0.55))
                    .foregroundColor(.white)
                    .clipShape(Circle())
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 16)
    }

    private func tag(text: String, color: Color, dot: Bool = false) -> some View {
        HStack(spacing: 6) {
            if dot {
                Circle().fill(color).frame(width: 8, height: 8)
            }
            Text(text)
                .font(.caption.monospacedDigit().bold())
                .foregroundColor(.white)
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(color.opacity(dot ? 0.35 : 0.85))
        .cornerRadius(8)
    }

    // MARK: Bottom controls

    private var bottomPanel: some View {
        VStack(spacing: 12) {
            statusLine
            actionButtons
        }
        .padding(16)
        .background(
            LinearGradient(
                colors: [Color.black.opacity(0.0), Color.black.opacity(0.85)],
                startPoint: .top, endPoint: .bottom)
        )
    }

    private var statusLine: some View {
        let text: String
        switch uploader.state {
        case .idle:
            text = recorder.status
        case let .uploading(sent, total):
            let pct = total > 0 ? Int(Double(sent) / Double(total) * 100) : 0
            text = "Uploading \(pct)%  \(formatBytes(sent)) / \(formatBytes(total))"
        case let .submitted(runId):
            text = "Submitted \(runId.prefix(8))…  Server processing."
        case let .completed(runId, _):
            text = "Done — \(runId.prefix(8)).  Tap VIEW."
        case let .failed(reason):
            text = "Failed: \(reason)"
        }
        return Text(text)
            .font(.callout)
            .foregroundColor(.white)
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
    }

    @ViewBuilder
    private var actionButtons: some View {
        switch (recorder.phase, uploader.state) {
        case (.initializing, _):
            ProgressView().tint(.white)

        case (.ready, _):
            recButton(label: "REC")

        case (.recording, _):
            recButton(label: "STOP")

        case (.stopped, .idle):
            VStack(spacing: 8) {
                if zipBytes > 0 {
                    Text("Bundle: \(formatBytes(zipBytes))")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                }
                bigButton(
                    label: zipBytes > 0
                        ? "UPLOAD (\(formatBytes(zipBytes)))" : "ZIP & UPLOAD",
                    color: .blue
                ) {
                    Task { await uploadCurrentSession() }
                }
                outlineButton(label: "RECORD AGAIN") {
                    zipBytes = 0
                    recorder.startRecording()
                }
            }

        case (_, .uploading(let sent, let total)):
            VStack(spacing: 8) {
                ProgressView(
                    value: total > 0 ? Double(sent) / Double(total) : 0
                )
                .tint(.blue)
                Text("\(formatBytes(sent)) / \(formatBytes(total))")
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.white.opacity(0.8))
            }

        case (_, .submitted):
            VStack(spacing: 8) {
                ProgressView().tint(.white)
                Text(
                    "Server processing — texture bake (~30 s) + 3DGS training "
                        + "(~5–7 min on a warm GPU). Cold-start adds 1–2 min."
                )
                .font(.caption)
                .foregroundColor(.white.opacity(0.7))
                .multilineTextAlignment(.center)
            }

        case (_, .completed(_, let url)):
            VStack(spacing: 8) {
                bigButton(label: "VIEW IN APP", color: .green) {
                    webURL = url
                    showWebView = true
                    log.log("opening webview → \(url.absoluteString)")
                }
                outlineButton(label: "COPY URL") {
                    UIPasteboard.general.string = url.absoluteString
                }
                outlineButton(label: "NEW CAPTURE") {
                    uploader.reset()
                    zipBytes = 0
                    recorder.startRecording()
                }
                Text(url.absoluteString)
                    .font(.caption2.monospaced())
                    .foregroundColor(.white.opacity(0.7))
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
            }

        case (_, .failed):
            outlineButton(label: "RESET") {
                uploader.reset()
            }
        }
    }

    private func recButton(label: String) -> some View {
        Button(action: {
            if case .recording = recorder.phase {
                recorder.stopRecording()
            } else {
                recorder.startRecording()
            }
        }) {
            HStack {
                Image(systemName: label == "STOP" ? "stop.circle.fill" : "circle.fill")
                Text(label).font(.title3.bold())
            }
            .frame(maxWidth: .infinity, minHeight: 56)
            .background(label == "STOP" ? Color.red : Color.red.opacity(0.85))
            .foregroundColor(.white)
            .cornerRadius(12)
        }
    }

    private func bigButton(label: String, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label).font(.title3.bold())
                .frame(maxWidth: .infinity, minHeight: 56)
                .background(color)
                .foregroundColor(.white)
                .cornerRadius(12)
        }
    }

    private func outlineButton(label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(label).font(.callout.bold())
                .frame(maxWidth: .infinity, minHeight: 44)
                .foregroundColor(.white)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.white.opacity(0.5), lineWidth: 1))
        }
    }

    // MARK: Actions

    private func uploadCurrentSession() async {
        guard case let .stopped(directory, _) = recorder.phase else { return }
        let logRef = log  // capture before hopping off the main actor
        do {
            let result = try await Task.detached(priority: .userInitiated) {
                try Bundler.zipSession(at: directory, log: logRef)
            }.value
            zippedBundle = result.url
            zipBytes = result.zipBytes
            await uploader.upload(zipURL: result.url)
        } catch {
            log.log("upload pipeline FAILED: \(error)")
            uploader.reset()
        }
    }
}

// MARK: - Log overlay

struct LogPanel: View {
    @ObservedObject var log: LogStream

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(log.entries) { entry in
                        HStack(alignment: .top, spacing: 6) {
                            Text(LogStream.formatter.string(from: entry.timestamp))
                                .foregroundColor(.white.opacity(0.4))
                            Text(entry.message)
                                .foregroundColor(.white.opacity(0.92))
                        }
                        .font(.system(size: 10, design: .monospaced))
                        .id(entry.id)
                    }
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color.black.opacity(0.78))
            .cornerRadius(8)
            .onChange(of: log.entries.count) { _, _ in
                if let last = log.entries.last {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }
}

// MARK: - Embedded mesh viewer
//
// SFSafariViewController gave a blank black screen — most likely because its
// captive WebKit instance handles ESM importmaps differently than full
// Safari, and our viewer page (/mesh, /splat, /viewer) loads three.js as
// modules. WKWebView with javaScriptEnabled gives us:
//  - explicit load + fail callbacks (logged to the on-screen LogStream)
//  - JS console capture so the viewer's status messages surface in our app
//  - the ability to dismiss with a button
struct ViewerWebView: View {
    let url: URL
    let log: LogStream
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Button("Close") { dismiss() }
                    .padding(.horizontal, 12)
                Spacer()
                Text(url.host ?? "")
                    .font(.caption.monospaced())
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer()
                Button("Open ↗") {
                    UIApplication.shared.open(url)
                }
                .padding(.horizontal, 12)
            }
            .frame(height: 44)
            .background(Color(white: 0.95))
            WebViewContainer(url: url, log: log)
        }
    }
}

private struct WebViewContainer: UIViewRepresentable {
    let url: URL
    let log: LogStream

    func makeCoordinator() -> Coordinator { Coordinator(log: log) }

    func makeUIView(context: Context) -> WKWebView {
        let cfg = WKWebViewConfiguration()
        // Surface viewer console.log() calls in our on-screen log stream.
        cfg.userContentController.add(context.coordinator, name: "lingbotLog")
        let consoleHook = WKUserScript(
            source: """
                (function() {
                    var orig = { log: console.log, warn: console.warn, error: console.error };
                    function send(level, args) {
                        try {
                            window.webkit.messageHandlers.lingbotLog.postMessage(
                                level + " " + Array.from(args).map(function(a){
                                    try { return typeof a === 'string' ? a : JSON.stringify(a); }
                                    catch(_) { return String(a); }
                                }).join(" ")
                            );
                        } catch (_) {}
                    }
                    console.log = function(){ send("log", arguments); orig.log.apply(console, arguments); };
                    console.warn = function(){ send("warn", arguments); orig.warn.apply(console, arguments); };
                    console.error = function(){ send("error", arguments); orig.error.apply(console, arguments); };
                    window.addEventListener("error", function(e) {
                        send("uncaught", [e.message + " @ " + e.filename + ":" + e.lineno]);
                    });
                })();
                """,
            injectionTime: .atDocumentStart,
            forMainFrameOnly: false
        )
        cfg.userContentController.addUserScript(consoleHook)
        let prefs = WKWebpagePreferences()
        prefs.allowsContentJavaScript = true
        cfg.defaultWebpagePreferences = prefs

        let webView = WKWebView(frame: .zero, configuration: cfg)
        webView.navigationDelegate = context.coordinator
        webView.backgroundColor = .black
        webView.isOpaque = false
        webView.load(URLRequest(url: url))
        log.log("webview load: \(url.absoluteString)")
        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        let log: LogStream
        init(log: LogStream) { self.log = log }

        func userContentController(
            _ userContentController: WKUserContentController,
            didReceive message: WKScriptMessage
        ) {
            if let text = message.body as? String {
                log.log("[js] \(text)")
            }
        }

        func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
            log.log("webview start")
        }
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            log.log("webview loaded")
        }
        func webView(
            _ webView: WKWebView,
            didFail navigation: WKNavigation!,
            withError error: Error
        ) {
            log.log("webview FAIL: \(error.localizedDescription)")
        }
        func webView(
            _ webView: WKWebView,
            didFailProvisionalNavigation navigation: WKNavigation!,
            withError error: Error
        ) {
            log.log("webview NAV FAIL: \(error.localizedDescription)")
        }
    }
}
