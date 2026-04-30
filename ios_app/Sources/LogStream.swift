import Foundation
import SwiftUI

/// Lightweight in-memory log buffer that's safe to write to from any thread.
///
/// Every component (recorder, bundler, uploader) writes here so the UI can
/// show what's actually happening without us needing Xcode attached.
@MainActor
final class LogStream: ObservableObject, @unchecked Sendable {
    struct Entry: Identifiable, Equatable {
        let id = UUID()
        let timestamp: Date
        let message: String
    }

    @Published private(set) var entries: [Entry] = []
    private let maxLines = 200
    nonisolated static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    /// Nonisolated so callers from any thread (AR delegate, Bundler, URLSession
    /// callbacks) can log without `await` ceremony — the actual @Published
    /// mutation is dispatched to main below.
    nonisolated func log(_ message: String) {
        let entry = Entry(timestamp: Date(), message: message)
        DispatchQueue.main.async { [weak self] in
            self?.append(entry)
        }
        print("[lingbot] \(LogStream.formatter.string(from: entry.timestamp)) \(message)")
    }

    func clear() {
        entries.removeAll()
    }

    func formatted() -> String {
        entries
            .map { "\(LogStream.formatter.string(from: $0.timestamp)) \($0.message)" }
            .joined(separator: "\n")
    }

    private func append(_ entry: Entry) {
        entries.append(entry)
        if entries.count > maxLines {
            entries.removeFirst(entries.count - maxLines)
        }
    }
}

/// Format a byte count as a short human-readable string (KB / MB / GB).
func formatBytes(_ bytes: Int64) -> String {
    let formatter = ByteCountFormatter()
    formatter.allowedUnits = [.useKB, .useMB, .useGB]
    formatter.countStyle = .binary
    formatter.includesUnit = true
    return formatter.string(fromByteCount: bytes)
}
