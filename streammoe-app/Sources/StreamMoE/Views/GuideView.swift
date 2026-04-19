import SwiftUI
import Foundation

// Loads a bundled Markdown file and renders it via SwiftUI's built-in
// AttributedString parser. Good enough for headings + inline code + links;
// for richer rendering the view can be swapped for a dependency later.
struct GuideView: View {
    let resource: String

    var body: some View {
        ScrollView {
            if let text = loadMarkdown(), let attr = try? AttributedString(markdown: text, options: .init(interpretedSyntax: .full)) {
                Text(attr)
                    .textSelection(.enabled)
                    .padding(18)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Text("Guide not bundled yet (\(resource).md).")
                    .foregroundStyle(.secondary)
                    .padding(18)
            }
        }
    }

    private func loadMarkdown() -> String? {
        // Look up Resources/Guides/<name>.md at the app's resource root. Falls
        // back to the process-relative path so `swift run` finds it during dev.
        if let url = Bundle.main.url(forResource: resource, withExtension: "md", subdirectory: "Guides") {
            return try? String(contentsOf: url, encoding: .utf8)
        }
        let devPath = "Resources/Guides/\(resource).md"
        if FileManager.default.fileExists(atPath: devPath) {
            return try? String(contentsOfFile: devPath, encoding: .utf8)
        }
        return nil
    }
}
