// Nous 读心助手 — reads the text of AI desktop apps (ChatGPT for now) through macOS Accessibility and hands it to the
// Nous colleague on this computer (127.0.0.1:7870/ax, with the local token). It does nothing else: no parsing, no network
// beyond 127.0.0.1, nothing written to disk except a one-off structure dump when run with --dump.
// Kept deliberately small and stable: macOS ties the Accessibility permission to this exact build, so all logic that may
// change lives in the Python colleague instead.
import Cocoa
import ApplicationServices

let home = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".nous-colleague")
// which apps to read: ~/.nous-colleague/reader-apps.json {"bundle id": "name"}, re-read every loop, so adding an app
// never needs a rebuild (and never resets the permission). Matching by app name works too ("name:ChatGPT").
func targets() -> [String: String] {
    if let d = try? Data(contentsOf: home.appendingPathComponent("reader-apps.json")),
       let m = try? JSONSerialization.jsonObject(with: d) as? [String: String], !m.isEmpty { return m }
    return ["com.openai.codex": "ChatGPT", "com.openai.chat": "ChatGPT"]
}
func running(_ key: String) -> NSRunningApplication? {
    if key.hasPrefix("name:") { return NSWorkspace.shared.runningApplications.first { $0.localizedName == String(key.dropFirst(5)) } }
    return NSRunningApplication.runningApplications(withBundleIdentifier: key).first
}

func attr(_ e: AXUIElement, _ a: String) -> AnyObject? {
    var v: AnyObject?
    return AXUIElementCopyAttributeValue(e, a as CFString, &v) == .success ? v : nil
}

func node(_ e: AXUIElement, _ depth: Int, _ budget: inout Int) -> [String: Any]? {
    if budget <= 0 || depth > 60 { return nil }
    budget -= 1
    var n: [String: Any] = [:]
    for (k, a) in [("r", "AXRole"), ("s", "AXSubrole"), ("i", "AXIdentifier"), ("d", "AXDescription"), ("t", "AXTitle"), ("h", "AXHelp")] {
        if let v = attr(e, a) as? String, !v.isEmpty { n[k] = v }
    }
    if let v = attr(e, "AXValue") as? String, !v.isEmpty { n["v"] = String(v.prefix(20000)) }
    if let f = attr(e, "AXFocused") as? Bool, f { n["f"] = 1 }
    if let kids = attr(e, "AXChildren") as? [AXUIElement], !kids.isEmpty {
        let c = kids.compactMap { node($0, depth + 1, &budget) }
        if !c.isEmpty { n["c"] = c }
    }
    return n
}

// Chromium/Electron apps (the ChatGPT desktop app among them) only build their web content's accessibility tree once an
// assistive client asks for it: set AXManualAccessibility (Electron) and AXEnhancedUserInterface (Chromium) on the app.
var enabled = Set<pid_t>()
func enableTree(_ app: AXUIElement, _ pid: pid_t) {
    if enabled.contains(pid) { return }
    AXUIElementSetAttributeValue(app, "AXManualAccessibility" as CFString, kCFBooleanTrue)
    AXUIElementSetAttributeValue(app, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
    enabled.insert(pid)
    Thread.sleep(forTimeInterval: 1.5)          // the tree is built asynchronously
}

func snapshot(_ pid: pid_t) -> [String: Any] {
    let app = AXUIElementCreateApplication(pid)
    enableTree(app, pid)
    var budget = 12000
    let wins = (attr(app, "AXWindows") as? [AXUIElement]) ?? []
    let front = NSWorkspace.shared.frontmostApplication?.processIdentifier == pid
    return ["front": front, "windows": wins.compactMap { node($0, 0, &budget) }]
}

func post(_ body: [String: Any]) {
    guard let tok = try? String(contentsOf: home.appendingPathComponent("local.token"), encoding: .utf8),
          let data = try? JSONSerialization.data(withJSONObject: body) else { return }
    var req = URLRequest(url: URL(string: "http://127.0.0.1:7870/ax")!)
    req.httpMethod = "POST"; req.httpBody = data; req.timeoutInterval = 5
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    req.setValue("Bearer " + tok.trimmingCharacters(in: .whitespacesAndNewlines), forHTTPHeaderField: "Authorization")
    let sem = DispatchSemaphore(value: 0)
    URLSession.shared.dataTask(with: req) { _, _, _ in sem.signal() }.resume()
    _ = sem.wait(timeout: .now() + 6)
}

let opts = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
if !AXIsProcessTrustedWithOptions(opts) {
    // macOS shows its own prompt that opens System Settings; keep polling until the person grants it
    while !AXIsProcessTrusted() { Thread.sleep(forTimeInterval: 3) }
}

if CommandLine.arguments.contains("--dump") {
    var out: [String: Any] = [:]
    for (key, name) in targets() {
        if let a = running(key) { out[name] = snapshot(a.processIdentifier) }
    }
    let url = home.appendingPathComponent("ax-dump.json")
    if let d = try? JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted]) {
        try? d.write(to: url); try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
    exit(0)
}

var last: [String: Int] = [:]
while true {
    var seen = Set<pid_t>()
    for (key, name) in targets() {
        guard let a = running(key), !seen.contains(a.processIdentifier) else { continue }
        seen.insert(a.processIdentifier)
        let snap = snapshot(a.processIdentifier)
        guard let d = try? JSONSerialization.data(withJSONObject: snap, options: [.sortedKeys]) else { continue }
        let h = d.hashValue
        if last[name] == h { continue }          // only when something on screen changed
        last[name] = h
        post(["app": name, "snapshot": snap])
    }
    Thread.sleep(forTimeInterval: 1.5)
}
