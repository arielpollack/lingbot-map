import Foundation

enum Backend {
    static let baseURL = URL(string: "https://arielpollack--lingbot-map-web-fastapi-app.modal.run")!

    static func runsBundleURL() -> URL {
        baseURL.appendingPathComponent("api/runs/bundle")
    }

    static func runStatusURL(runId: String) -> URL {
        baseURL.appendingPathComponent("api/runs/\(runId)")
    }

    static func meshViewURL(runId: String) -> URL {
        baseURL.appendingPathComponent("mesh").appending(queryItems: [URLQueryItem(name: "run", value: runId)])
    }

    static func splatViewURL(runId: String) -> URL {
        baseURL.appendingPathComponent("splat").appending(queryItems: [URLQueryItem(name: "run", value: runId)])
    }

    static func pointsViewURL(runId: String) -> URL {
        baseURL.appendingPathComponent("viewer").appending(queryItems: [URLQueryItem(name: "run", value: runId)])
    }
}

private extension URL {
    func appending(queryItems: [URLQueryItem]) -> URL {
        var c = URLComponents(url: self, resolvingAgainstBaseURL: false)!
        c.queryItems = (c.queryItems ?? []) + queryItems
        return c.url!
    }
}
