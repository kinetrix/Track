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
                Self.wire(camera: cam, pipeline: p)
                Task { @MainActor in
                    p.onResult = { [weak self] out in  // delivered on the main queue
                        guard let self else { return }
                        self.faces = self.facesOn ? out.faces : []
                        self.hands = self.handsOn ? out.hands : []
                        self.stats = out.stats
                        self.frameSize = out.size
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
                Overlay(faces: model.faces, hands: model.hands, frameSize: model.frameSize, showMasks: model.showMasks)
                HUD(stats: model.stats).padding(8)
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
                Toggle("Mirror", isOn: $model.mirrored)
                Toggle("Masks", isOn: $model.showMasks)
                Toggle("Faces", isOn: $model.facesOn)
                Toggle("Hands", isOn: $model.handsOn)
                Button { model.reset() } label: { Label("Reset", systemImage: "arrow.counterclockwise") }
                Spacer()
            }
            .toggleStyle(.checkbox)
            .padding(10)
        }
        .frame(minWidth: 800, minHeight: 520)
        .onAppear { model.start() }
    }
}

struct HUD: View {
    let stats: PipelineStats
    var body: some View {
        Text(String(format: "%.1f FPS · %ld face%@ · %ld hand%@\ndetect %.1f · encode %.1f · per face %.1f · palm %.1f · landmarks %.1f · total %.1f ms",
                    stats.fps, stats.faces, stats.faces == 1 ? "" : "s", stats.hands, stats.hands == 1 ? "" : "s",
                    stats.detectMs, stats.encodeMs, stats.perFaceMs, stats.palmMs, stats.landmarksMs, stats.totalMs)
             + (stats.events.isEmpty ? "" : "\n" + stats.events.joined(separator: "\n")))
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
                for f in faces {
                    let color = Self.color(f.id)
                    if showMasks, let m = f.mask {
                        ctx.draw(Image(decorative: m, scale: 1), in: CGRect(x: ox, y: oy, width: w, height: h))
                    }
                    let r = CGRect(x: ox + f.box.minX * w, y: oy + f.box.minY * h, width: f.box.width * w, height: f.box.height * h)
                    if f.visible {
                        ctx.stroke(Path(roundedRect: r, cornerRadius: 6), with: .color(color), lineWidth: 3)
                    } else {
                        ctx.stroke(Path(roundedRect: r, cornerRadius: 6), with: .color(color.opacity(0.5)),
                                   style: StrokeStyle(lineWidth: 2, dash: [6, 6]))
                    }
                    ctx.draw(Text("face #\(f.id)\(f.visible ? "" : " (hidden)")").font(.headline).foregroundColor(color),
                             at: CGPoint(x: r.minX, y: r.minY - 12), anchor: .leading)
                }
                for hand in hands {
                    let color = Self.color(hand.id + 2)
                    let pts = hand.points.map(map)
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
                    let label = "\(hand.handedness) #\(hand.id): \(hand.gesture.isEmpty ? "–" : hand.gesture) (\(hand.fingers))"
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
