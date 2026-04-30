import Foundation

@MainActor
final class Uploader: NSObject, ObservableObject {
    enum UploadState: Equatable {
        case idle
        case uploading(bytesSent: Int64, bytesTotal: Int64)
        case submitted(runId: String)
        case completed(runId: String, viewerURL: URL)
        case failed(reason: String)
    }

    @Published private(set) var state: UploadState = .idle
    let log: LogStream

    private var pollTask: Task<Void, Never>? = nil
    private var session: URLSession?

    init(log: LogStream) {
        self.log = log
        super.init()
    }

    deinit {
        pollTask?.cancel()
        session?.invalidateAndCancel()
    }

    /// Build the body file, send the multipart upload via a delegate-backed
    /// URLSession (so we get real `didSendBodyData` progress), and start
    /// polling for the run result.
    func upload(zipURL: URL) async {
        let zipBytes =
            ((try? FileManager.default.attributesOfItem(atPath: zipURL.path))?[.size]
                as? NSNumber)?.int64Value ?? 0
        state = .uploading(bytesSent: 0, bytesTotal: zipBytes)
        log.log("upload start; \(formatBytes(zipBytes)) zip → /api/runs/bundle")

        do {
            let runId = try await postBundle(zipURL: zipURL, zipBytes: zipBytes)
            log.log("server accepted bundle; run_id=\(runId)")
            state = .submitted(runId: runId)
            startPolling(runId: runId)
        } catch {
            log.log("upload FAILED: \(error)")
            state = .failed(reason: "\(error)")
        }
    }

    func reset() {
        pollTask?.cancel()
        pollTask = nil
        state = .idle
    }

    // MARK: Multipart POST

    private func postBundle(zipURL: URL, zipBytes: Int64) async throws -> String {
        let boundary = "----LingbotBoundary\(UUID().uuidString)"
        var request = URLRequest(url: Backend.runsBundleURL())
        request.httpMethod = "POST"
        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        request.timeoutInterval = 1800  // 30 min cap for very large captures

        // Stream the multipart body to a temp file. Avoids loading the whole
        // zip into RAM — important on phones with limited memory and bundles
        // that can run hundreds of MB.
        let bodyURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("upload-\(UUID().uuidString).bin")
        try? FileManager.default.removeItem(at: bodyURL)
        FileManager.default.createFile(atPath: bodyURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: bodyURL)

        let zipName = zipURL.lastPathComponent
        let header =
            "--\(boundary)\r\n"
            + "Content-Disposition: form-data; name=\"bundle\"; filename=\"\(zipName)\"\r\n"
            + "Content-Type: application/zip\r\n\r\n"
        try handle.write(contentsOf: header.data(using: .utf8)!)

        let zipHandle = try FileHandle(forReadingFrom: zipURL)
        var copied: Int64 = 0
        let chunkSize = 1 << 20  // 1 MiB
        while true {
            let chunk = zipHandle.readData(ofLength: chunkSize)
            if chunk.isEmpty { break }
            try handle.write(contentsOf: chunk)
            copied += Int64(chunk.count)
        }
        try zipHandle.close()

        let footer = "\r\n--\(boundary)--\r\n"
        try handle.write(contentsOf: footer.data(using: .utf8)!)
        try handle.close()

        let bodyBytes =
            ((try? FileManager.default.attributesOfItem(atPath: bodyURL.path))?[.size]
                as? NSNumber)?.int64Value ?? 0
        log.log("multipart body built; \(formatBytes(bodyBytes)) → POST")

        // Build the URLSession lazily so the delegate gets wired up to *this*
        // Uploader instance. Keep it around so the delegate stays alive until
        // we deinit.
        let urlSession = makeSession()
        let (data, response) = try await urlSession.upload(for: request, fromFile: bodyURL)
        try? FileManager.default.removeItem(at: bodyURL)

        guard let http = response as? HTTPURLResponse else {
            throw UploadError.invalidResponse
        }
        log.log("HTTP \(http.statusCode) (\(data.count) bytes response)")
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? "<binary>"
            throw UploadError.httpStatus(http.statusCode, body)
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let runId = json["id"] as? String
        else {
            throw UploadError.malformedResponse
        }
        return runId
    }

    private func makeSession() -> URLSession {
        if let s = session { return s }
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 1800
        cfg.timeoutIntervalForResource = 1800
        cfg.waitsForConnectivity = true
        let s = URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
        session = s
        return s
    }

    // MARK: Polling

    private func startPolling(runId: String) {
        pollTask?.cancel()
        log.log("polling /api/runs/\(runId.prefix(8))…")
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                guard let self = self else { return }
                await self.pollOnce(runId: runId)
                if case .completed = self.state { return }
                if case .failed = self.state { return }
            }
        }
    }

    private func pollOnce(runId: String) async {
        var request = URLRequest(url: Backend.runStatusURL(runId: runId))
        request.timeoutInterval = 30
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                return
            }
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return }
            let status = (json["status"] as? String) ?? "unknown"
            log.log("poll status=\(status)")
            switch status {
            case "completed":
                let output = json["output"] as? [String: Any]
                let url = bestViewerURL(runId: runId, output: output)
                state = .completed(runId: runId, viewerURL: url)
                log.log("DONE → \(url.absoluteString)")
            case "failed":
                let reason = (json["error"] as? String) ?? "failed"
                state = .failed(reason: reason)
            default:
                break
            }
        } catch {
            log.log("poll error (transient): \(error.localizedDescription)")
        }
    }

    private func bestViewerURL(runId: String, output: [String: Any]?) -> URL {
        // Splat first — it's the photoreal view, and any run that produced
        // both mesh.glb and splat.ply did so because the user wanted the
        // photoreal output (the lidar_mesh tier ships both).
        if let splat = output?["splat_key"] as? String, !splat.isEmpty {
            return Backend.splatViewURL(runId: runId)
        }
        if let mesh = output?["mesh_key"] as? String, !mesh.isEmpty {
            return Backend.meshViewURL(runId: runId)
        }
        return Backend.pointsViewURL(runId: runId)
    }
}

// MARK: - URLSession delegate (real upload progress)

extension Uploader: URLSessionTaskDelegate {
    nonisolated func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didSendBodyData bytesSent: Int64,
        totalBytesSent: Int64,
        totalBytesExpectedToSend: Int64
    ) {
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            // Cap at the actual body size so the bar doesn't briefly show
            // 100% while we're still waiting for the response.
            self.state = .uploading(
                bytesSent: totalBytesSent,
                bytesTotal: max(totalBytesExpectedToSend, totalBytesSent)
            )
            // Log every ~10% so we can see progress without spam.
            if totalBytesExpectedToSend > 0 {
                let pct = Double(totalBytesSent) / Double(totalBytesExpectedToSend) * 100
                let nearestTen = Int(pct / 10) * 10
                if nearestTen != self.lastLoggedUploadPct, nearestTen >= 10 {
                    self.lastLoggedUploadPct = nearestTen
                    self.log.log(
                        "upload \(nearestTen)% (\(formatBytes(totalBytesSent)))"
                    )
                }
            }
        }
    }
}

private extension Uploader {
    var lastLoggedUploadPct: Int {
        get { (objc_getAssociatedObject(self, &Uploader.lastPctKey) as? Int) ?? 0 }
        set { objc_setAssociatedObject(self, &Uploader.lastPctKey, newValue, .OBJC_ASSOCIATION_RETAIN) }
    }
    static var lastPctKey: UInt8 = 0
}

enum UploadError: LocalizedError {
    case invalidResponse
    case malformedResponse
    case httpStatus(Int, String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse: return "Server returned an invalid response."
        case .malformedResponse: return "Server response missing run id."
        case let .httpStatus(code, body): return "HTTP \(code): \(body.prefix(200))"
        }
    }
}
