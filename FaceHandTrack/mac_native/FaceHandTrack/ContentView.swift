import AppKit
import AVFoundation
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    @Published var faces: [FaceTrackOutput] = []
    @Published var hands: [HandTrackOutput] = []
    @Published var stats = PipelineStats()
    @Published var frameSize = CGSize(width: 1280, height: 720)
    @Published var error: String?
    @Published var loading = true
    @Published var cameras: [AVCaptureDevice] = []
    @Published var cameraID: String = "" {
        didSet { if cameraID != oldValue { switchCamera() } }
    }
    @Published var mirrored = true {
        didSet {
            camera.setMirrored(mirrored)
            realsense?.mirrored = mirrored
            pipeline?.mirrored = mirrored
            pipeline?.reset()
        }
    }
    /// The RealSense colour stream through Intel's SDK is the current source (needs sudo on macOS).
    @Published var usingRealSense = false
    @Published var realsenseName = ""
    static let realsenseID = "realsense-sdk"
    @Published var showMasks = true { didSet { pipeline?.renderMasks = showMasks } }
    @Published var facesOn = true { didSet { pipeline?.facesEnabled = facesOn; if !facesOn { faces = [] } } }
    @Published var handsOn = true { didSet { pipeline?.handsEnabled = handsOn; if !handsOn { hands = [] } } }
    @Published var faceMode = FaceMode.fast { didSet { if faceMode != oldValue { pipeline?.faceMode = faceMode } } }
    @Published var alerts: [HelpAlert] = []
    @Published var attention: AttentionTarget?
    private var lastAlertID = 0
    @Published var nameText = ""
    @Published var namesStatus = ""

    func saveFace() {
        pipeline?.saveFace(name: nameText) { [weak self] msg in
            Task { @MainActor in
                self?.namesStatus = msg
                if msg.hasPrefix("Saved") { self?.nameText = "" }
            }
        }
    }

    func acknowledgeAlerts() {
        pipeline?.acknowledgeAlerts()
        alerts = []
    }

    func forgetNames() {
        pipeline?.forgetNames { [weak self] msg in Task { @MainActor in self?.namesStatus = msg } }
    }

    let camera = CameraController()
    let frameSink = FrameSink()
    private var pipeline: Pipeline?
    private var realsense: RealSenseColor?
    /// Started with --realsense ("Run with RealSense.command"): use the RealSense colour stream.
    let wantsRealSense = CommandLine.arguments.contains("--realsense")

    func start() {
        guard pipeline == nil else { return }
        cameras = CameraController.cameras()
        let first = cameras.first
        cameraID = wantsRealSense ? Self.realsenseID : (first?.uniqueID ?? "")
        print("[camera] found: " + cameras.map(\.localizedName).joined(separator: ", "))
        let cam = camera
        let mirror = mirrored
        // The first launch compiles the models for this Mac (a few seconds): load off the main thread.
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let p = try Pipeline()
                p.mirrored = mirror
                p.faceMode = .fast
                Self.wire(camera: cam, pipeline: p)
                Task { @MainActor in
                    p.onResult = { [weak self] out in  // delivered on the main queue
                        guard let self else { return }
                        self.faces = self.facesOn ? out.faces : []
                        self.hands = self.handsOn ? out.hands : []
                        self.stats = out.stats
                        self.frameSize = out.size
                        self.alerts = out.alerts
                        self.attention = out.attention
                        if let newest = out.alerts.last, newest.id > self.lastAlertID {  // a new help request: sound
                            self.lastAlertID = newest.id
                            if NSSound(named: NSSound.Name("Sosumi"))?.play() != true { NSSound.beep() }
                        }
                    }
                    self.pipeline = p
                    self.loading = false
                    if self.wantsRealSense {
                        self.startRealSense()
                        return
                    }
                    AVCaptureDevice.requestAccess(for: .video) { ok in
                        if ok {
                            cam.start(device: first, mirrored: mirror)
                        } else {
                            Task { @MainActor in
                                self.error = "Camera access is off. Allow FaceHandTrack in System Settings > Privacy & Security > Camera."
                            }
                        }
                    }
                }
            } catch {
                let msg = "Model loading failed: \(error.localizedDescription)"
                Task { @MainActor in
                    self.error = msg
                    self.loading = false
                }
            }
        }
    }

    /// Camera frames arrive on the camera queue; keep this closure off the main actor.
    nonisolated private static func wire(camera: CameraController, pipeline: Pipeline) {
        camera.onFrame = { [weak pipeline] pb in pipeline?.submit(pb) }
    }

    /// RealSense frames arrive on its capture thread; keep this closure off the main actor too.
    nonisolated private static func wire(realsense: RealSenseColor, pipeline: Pipeline, sink: FrameSink) {
        realsense.onFrame = { [weak pipeline] pb in
            pipeline?.submit(pb)
            sink.show(pb)
        }
    }

    private func switchCamera() {
        guard pipeline != nil else { return }
        pipeline?.reset()
        if cameraID == Self.realsenseID {
            startRealSense()
            return
        }
        guard let d = cameras.first(where: { $0.uniqueID == cameraID }) else { return }
        realsense?.stop()
        usingRealSense = false
        camera.start(device: d, mirrored: mirrored)
    }

    /// RealSense colour stream via librealsense; frames go to the pipeline and to the preview layer.
    private func startRealSense() {
        guard let p = pipeline else { return }
        camera.stop()
        do {
            let rs = try realsense ?? RealSenseColor()
            realsense = rs
            rs.mirrored = mirrored
            Self.wire(realsense: rs, pipeline: p, sink: frameSink)
            try rs.start()
            realsenseName = rs.deviceDescription
            usingRealSense = true
            error = nil
        } catch {
            usingRealSense = false
            self.error = "RealSense: \(error.localizedDescription)"
            print("[realsense] " + (self.error ?? ""))
        }
    }

    func reset() { pipeline?.reset() }
}

struct ContentView: View {
    @StateObject private var model = AppModel()

    var body: some View {
        VStack(spacing: 0) {
            ZStack(alignment: .topLeading) {
                Color.black
                if model.usingRealSense {
                    FramePreview(sink: model.frameSink)
                } else {
                    CameraPreview(camera: model.camera, mirrored: model.mirrored)
                }
                Overlay(faces: model.faces, hands: model.hands, frameSize: model.frameSize,
                        showMasks: model.showMasks && model.faceMode == .edgetam)
                HUD(stats: model.stats, mode: model.faceMode).padding(8)
                if let a = model.attention {
                    AttentionBadge(target: a, name: model.faces.first { $0.id == a.faceID }?.name)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
                        .padding(.bottom, 14)
                }
                if !model.alerts.isEmpty {
                    HelpBanner(alerts: model.alerts) { model.acknowledgeAlerts() }
                        .frame(maxWidth: .infinity, alignment: .top)
                        .padding(.top, 12)
                }
                if model.loading {
                    ProgressView("Loading models…").controlSize(.large).frame(maxWidth: .infinity, maxHeight: .infinity)
                        .foregroundStyle(.white)
                }
                if let e = model.error {
                    Text(e).padding().background(.red.opacity(0.85)).foregroundStyle(.white)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            HStack(spacing: 16) {
                Picker("Camera", selection: $model.cameraID) {
                    if model.wantsRealSense {
                        Text("Intel RealSense colour (SDK)").tag(AppModel.realsenseID)
                    }
                    ForEach(model.cameras, id: \.uniqueID) { Text($0.localizedName).tag($0.uniqueID) }
                }
                .frame(maxWidth: 360)
                Picker("Face tracker", selection: $model.faceMode) {
                    ForEach(FaceMode.allCases) { Text($0.rawValue).tag($0) }
                }
                .frame(maxWidth: 260)
                Toggle("Mirror", isOn: $model.mirrored)
                if model.faceMode == .edgetam { Toggle("Masks", isOn: $model.showMasks) }
                Toggle("Faces", isOn: $model.facesOn)
                Toggle("Hands", isOn: $model.handsOn)
                Button { model.reset() } label: { Label("Reset", systemImage: "arrow.counterclockwise") }
                Spacer()
            }
            .toggleStyle(.checkbox)
            .padding(.horizontal, 10)
            .padding(.top, 10)
            HStack(spacing: 10) {
                TextField("Name", text: $model.nameText)
                    .frame(maxWidth: 180)
                    .onSubmit { model.saveFace() }
                Button("Save face") { model.saveFace() }
                    .help("Saves the fingerprint of the person addressing the camera, else the largest face")
                Button("Forget names") { model.forgetNames() }
                Text(model.namesStatus).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Spacer()
            }
            .padding(10)
        }
        .frame(minWidth: 800, minHeight: 520)
        .onAppear { model.start() }
    }
}

/// Who the robot is attending to and how it should turn to face them.
struct AttentionBadge: View {
    let target: AttentionTarget
    let name: String?

    var body: some View {
        let who = name.map { "#\(target.faceID) \($0)" } ?? "face #\(target.faceID)"
        Text("Robot → \(who) (\(target.reason)) · \(target.turnText)")
            .font(.title3.bold())
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(.cyan.opacity(0.85), in: Capsule())
            .foregroundStyle(.black)
    }
}

/// Red banner for unacknowledged help requests, with the button that clears them.
struct HelpBanner: View {
    let alerts: [HelpAlert]
    let acknowledge: () -> Void

    static let timeFormat: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("HELP REQUESTED").font(.title.bold())
            ForEach(alerts.suffix(3)) { a in
                Text("\(a.who) · \(a.reason) · \(Self.timeFormat.string(from: a.clock))").font(.title3)
            }
            Button("Acknowledge", action: acknowledge).controlSize(.large)
        }
        .padding(14)
        .background(.red.opacity(0.9), in: RoundedRectangle(cornerRadius: 12))
        .foregroundStyle(.white)
    }
}

struct HUD: View {
    let stats: PipelineStats
    let mode: FaceMode

    var line: String {
        let counts = String(format: "%.1f FPS · %ld face%@ · %ld hand%@", stats.fps, stats.faces, stats.faces == 1 ? "" : "s",
                            stats.hands, stats.hands == 1 ? "" : "s")
        let faces = mode == .fast
            ? String(format: "detect %.1f · fingerprints %.1f", stats.detectMs, stats.embedMs)
            : String(format: "detect %.1f · encode %.1f · per face %.1f", stats.detectMs, stats.encodeMs, stats.perFaceMs)
        let rest = String(format: " · palm %.1f · landmarks %.1f · total %.1f ms", stats.palmMs, stats.landmarksMs, stats.totalMs)
        return counts + "\n" + faces + rest + (stats.events.isEmpty ? "" : "\n" + stats.events.joined(separator: "\n"))
    }

    var body: some View {
        Text(line)
            .font(.caption.monospacedDigit())
            .padding(8)
            .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))
            .foregroundStyle(.white)
    }
}

/// Face masks/boxes and hand skeletons, with the same aspect-fit mapping as the preview.
struct Overlay: View {
    let faces: [FaceTrackOutput]
    let hands: [HandTrackOutput]
    let frameSize: CGSize
    let showMasks: Bool

    static let bones = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                        (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

    static func color(_ i: Int) -> Color {
        let (r, g, b) = MaskRenderer.palette[i % MaskRenderer.palette.count]
        return Color(red: Double(r) / 255, green: Double(g) / 255, blue: Double(b) / 255)
    }

    var body: some View {
        GeometryReader { geo in
            let s = min(geo.size.width / frameSize.width, geo.size.height / frameSize.height)
            let w = frameSize.width * s, h = frameSize.height * s
            let ox = (geo.size.width - w) / 2, oy = (geo.size.height - h) / 2
            let map = { (p: CGPoint) in CGPoint(x: ox + p.x * w, y: oy + p.y * h) }
            Canvas { ctx, _ in
                var faceRects: [Int: CGRect] = [:]
                for f in faces {
                    let color = Self.color(f.id)
                    if showMasks, let m = f.mask {
                        ctx.draw(Image(decorative: m, scale: 1), in: CGRect(x: ox, y: oy, width: w, height: h))
                    }
                    let r = CGRect(x: ox + f.box.minX * w, y: oy + f.box.minY * h, width: f.box.width * w, height: f.box.height * h)
                    faceRects[f.id] = r
                    if f.addressing != nil {  // the person talking to the camera
                        ctx.stroke(Path(roundedRect: r.insetBy(dx: -6, dy: -6), cornerRadius: 10), with: .color(.yellow), lineWidth: 5)
                    }
                    if f.visible {
                        ctx.stroke(Path(roundedRect: r, cornerRadius: 6), with: .color(color), lineWidth: 3)
                    } else {
                        ctx.stroke(Path(roundedRect: r, cornerRadius: 6), with: .color(color.opacity(0.5)),
                                   style: StrokeStyle(lineWidth: 2, dash: [6, 6]))
                    }
                    var label = f.name.map { "#\(f.id) \($0)" } ?? "face #\(f.id)"
                    if !f.visible { label += " (hidden)" } else if f.facing == false { label += " · looking away" }
                    if f.waving { label += " · waving" }
                    ctx.draw(Text(label).font(.headline).foregroundColor(color),
                             at: CGPoint(x: r.minX, y: r.minY - 12), anchor: .leading)
                    if let g = f.addressing {
                        let banner = ctx.resolve(Text("talking: \(g)").font(.title2.bold()).foregroundColor(.black))
                        let size = banner.measure(in: CGSize(width: 1000, height: 100))
                        let origin = CGPoint(x: r.minX, y: r.maxY + 10)
                        ctx.fill(Path(roundedRect: CGRect(origin: origin, size: size).insetBy(dx: -6, dy: -3), cornerRadius: 5),
                                 with: .color(.yellow))
                        ctx.draw(banner, at: origin, anchor: .topLeading)
                    }
                }
                for hand in hands {
                    // a hand takes its person's colour; grey when it belongs to no one
                    let color = hand.owner.map { Self.color($0) } ?? .gray
                    let pts = hand.points.map(map)
                    if let o = hand.owner, let fr = faceRects[o], let wrist = pts.first {  // hand -> its face
                        var link = Path()
                        link.move(to: wrist)
                        link.addLine(to: CGPoint(x: fr.midX, y: fr.maxY))
                        ctx.stroke(link, with: .color(color.opacity(0.7)), style: StrokeStyle(lineWidth: 2, dash: [4, 6]))
                    }
                    var bones = Path()
                    for (a, b) in Self.bones {
                        bones.move(to: pts[a])
                        bones.addLine(to: pts[b])
                    }
                    ctx.stroke(bones, with: .color(color), lineWidth: 3)
                    for p in pts {
                        ctx.fill(Path(ellipseIn: CGRect(x: p.x - 3, y: p.y - 3, width: 6, height: 6)), with: .color(.white))
                    }
                    let top = map(CGPoint(x: hand.box.minX, y: hand.box.minY))
                    let who = hand.owner.map { "#\($0)" } ?? "?"
                    let label = "\(who) · \(hand.handedness): \(hand.gesture.isEmpty ? "–" : hand.gesture) (\(hand.fingers))"
                    let text = ctx.resolve(Text(label).font(.title3.bold()).foregroundColor(color))
                    let size = text.measure(in: CGSize(width: 1000, height: 100))
                    let origin = CGPoint(x: max(0, top.x), y: max(0, top.y - size.height - 6))
                    ctx.fill(Path(roundedRect: CGRect(origin: origin, size: size).insetBy(dx: -5, dy: -3), cornerRadius: 4),
                             with: .color(.black.opacity(0.7)))
                    ctx.draw(text, at: origin, anchor: .topLeading)
                }
            }
        }
    }
}

/// AVCaptureVideoPreviewLayer in an NSView: smooth full-rate video under the overlay.
struct CameraPreview: NSViewRepresentable {
    let camera: CameraController
    let mirrored: Bool

    final class PreviewView: NSView {
        let previewLayer = AVCaptureVideoPreviewLayer()

        override init(frame: NSRect) {
            super.init(frame: frame)
            layer = previewLayer  // layer-hosting view: set the layer first, then wantsLayer
            wantsLayer = true
            previewLayer.videoGravity = .resizeAspect
            previewLayer.backgroundColor = NSColor.black.cgColor
        }

        required init?(coder: NSCoder) { fatalError("not used") }
    }

    func makeNSView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = camera.session
        camera.previewLayer = v.previewLayer
        camera.setMirrored(mirrored)
        return v
    }

    func updateNSView(_ nsView: PreviewView, context: Context) {}
}

/// Shows frames that do not come from AVFoundation (the RealSense SDK) on a layer.
final class FrameSink {
    weak var layer: CALayer?
    private var current: CVPixelBuffer?  // keeps the shown buffer alive until the next one

    /// Any thread.
    func show(_ pb: CVPixelBuffer) {
        DispatchQueue.main.async {
            guard let layer = self.layer, let surface = CVPixelBufferGetIOSurface(pb) else { return }
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            layer.contents = surface.takeUnretainedValue()
            CATransaction.commit()
            self.current = pb
        }
    }
}

struct FramePreview: NSViewRepresentable {
    let sink: FrameSink

    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        let layer = CALayer()
        layer.contentsGravity = .resizeAspect
        layer.backgroundColor = NSColor.black.cgColor
        v.layer = layer
        v.wantsLayer = true
        sink.layer = layer
        return v
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
