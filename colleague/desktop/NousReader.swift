// Nous 读心助手 — reads the text of AI desktop apps (ChatGPT for now) through macOS Accessibility and hands it to the
// Nous colleague on this computer (127.0.0.1:7870/ax, with the local token). It does nothing else: no parsing, no network
// beyond 127.0.0.1, nothing written to disk except a one-off structure dump when run with --dump.
// Kept deliberately small and stable: macOS ties the Accessibility permission to this exact build, so all logic that may
// change lives in the Python colleague instead.
import Cocoa
import ApplicationServices

let apps = ["com.openai.chat": "ChatGPT"]
let home = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".nous-colleague")

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

func snapshot(_ pid: pid_t) -> [String: Any] {
    let app = AXUIElementCreateApplication(pid)
    var budget = 8000
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
    for (bid, name) in apps {
        if let a = NSRunningApplication.runningApplications(withBundleIdentifier: bid).first { out[name] = snapshot(a.processIdentifier) }
    }
    let url = home.appendingPathComponent("ax-dump.json")
    if let d = try? JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted]) {
        try? d.write(to: url); try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
    exit(0)
}

var last: [String: Int] = [:]
while true {
    for (bid, name) in apps {
        guard let a = NSRunningApplication.runningApplications(withBundleIdentifier: bid).first else { continue }
        let snap = snapshot(a.processIdentifier)
        guard let d = try? JSONSerialization.data(withJSONObject: snap, options: [.sortedKeys]) else { continue }
        let h = d.hashValue
        if last[name] == h { continue }          // only when something on screen changed
        last[name] = h
        post(["app": name, "snapshot": snap])
    }
    Thread.sleep(forTimeInterval: 1.5)
}
