import SwiftUI
import AppKit

// Click to copy `value` to pasteboard, shows a transient "Copied" badge.
// Label is the user-facing name of the value (e.g. "Base URL"); the button
// text itself is the value so the user can visually confirm what will hit
// their clipboard.
struct CopyButton: View {
    let label: String
    let value: String
    @State private var didFlash = false

    var body: some View {
        HStack(spacing: 10) {
            Text(label)
                .font(.system(.caption, design: .default))
                .foregroundStyle(.secondary)
                .frame(width: 110, alignment: .leading)
            Button(action: copy) {
                HStack(spacing: 6) {
                    Text(value.isEmpty ? "—" : value)
                        .font(.system(.body, design: .monospaced))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 4)
                    Image(systemName: didFlash ? "checkmark.circle.fill" : "doc.on.doc")
                        .foregroundStyle(didFlash ? .green : .secondary)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color(nsColor: .controlBackgroundColor))
                )
            }
            .buttonStyle(.plain)
            .disabled(value.isEmpty)
        }
    }

    private func copy() {
        guard !value.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
        withAnimation(.easeInOut(duration: 0.15)) { didFlash = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            withAnimation(.easeInOut(duration: 0.4)) { didFlash = false }
        }
    }
}
