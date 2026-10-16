import AVFoundation
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    @Published var faces: [FaceTrackOutput] = []
    @Published var stats = PipelineStats()
    @Published var frameSize = CGSize(width: 720, height: 1280)
    @Published var error: String?
    @Published var showMasks = true {
        didSet { pipeline?.renderMasks = showMasks }
    }

    let camera = CameraController()
    private var pipeline: Pipeline?

    func start() {
        guard pipeline == nil else { return }
        let cam = camera
        // First launch compiles the models for the Neural Engine (can take a few
        // seconds), so load off the main thread.
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let p = try Pipeline()
                Self.wire(camera: cam, pipeline: p)
                Task { @MainActor in
                    p.onResult = { [weak self] faces, stats, size in  // delivered on the main queue
                        self?.faces = faces
                        self?.stats = stats
                        self?.frameSize = size
                    }
                    self.pipeline = p
                    AVCaptureDevice.requestAccess(for: .video) { ok in
                        if ok { cam.start() }
                    }
                }
            } catch {
                let msg = "Model loading failed: \(error.localizedDescription)"
                Task { @MainActor in self.error = msg }
            }
        }
    }

    /// Camera frames arrive on the camera queue; keep this closure off the main actor.
    nonisolated private static func wire(camera: CameraController, pipeline: Pipeline) {
        camera.onFrame = { [weak pipeline] pb in pipeline?.submit(pb) }
    }

    func flip() {
        pipeline?.reset()
        camera.flip()
    }

    func reset() { pipeline?.reset() }
}

struct ContentView: View {
    @StateObject private var model = AppModel()

    var body: some View {
        ZStack {
            CameraPreview(camera: model.camera).ignoresSafeArea()
            FaceOverlay(faces: model.faces, frameSize: model.frameSize, showMasks: model.showMasks).ignoresSafeArea()
            VStack {
                HUD(stats: model.stats)
                Spacer()
                HStack(spacing: 16) {
                    Button { model.flip() } label: { Label("Flip", systemImage: "arrow.triangle.2.circlepath.camera") }
                    Button { model.showMasks.toggle() } label: { Label(model.showMasks ? "Boxes" : "Masks", systemImage: "square.on.circle") }
                    Button { model.reset() } label: { Label("Reset", systemImage: "arrow.counterclockwise") }
                }
                .buttonStyle(.borderedProminent)
                .padding(.bottom, 24)
            }
            if let e = model.error {
                Text(e).padding().background(.red.opacity(0.8)).foregroundStyle(.white)
            }
        }
        .onAppear { model.start() }
    }
}

struct HUD: View {
    let stats: PipelineStats
    var body: some View {
        Text(String(format: "%.1f fps · %ld face%@\ndetect %.1f ms · encode %.1f ms · per face %.1f ms · total %.1f ms",
                    stats.fps, stats.faces, stats.faces == 1 ? "" : "s",
                    stats.detectMs, stats.encodeMs, stats.perFaceMs, stats.totalMs)
             + (stats.events.isEmpty ? "" : "\n" + stats.events.joined(separator: "\n")))
            .font(.caption.monospacedDigit())
            .padding(8)
            .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
            .foregroundStyle(.white)
            .padding(.top, 8)
    }
}

/// Draws masks/boxes with the same aspect-fill mapping as the preview layer.
struct FaceOverlay: View {
    let faces: [FaceTrackOutput]
    let frameSize: CGSize
    let showMasks: Bool

    var body: some View {
        GeometryReader { geo in
            let s = max(geo.size.width / frameSize.width, geo.size.height / frameSize.height)
            let w = frameSize.width * s, h = frameSize.height * s
            let ox = (geo.size.width - w) / 2, oy = (geo.size.height - h) / 2
            Canvas { ctx, _ in
                for f in faces {
                    let color = Color(uiColor: UIColor(red: CGFloat(MaskRenderer.palette[f.id % MaskRenderer.palette.count].0) / 255,
                                                       green: CGFloat(MaskRenderer.palette[f.id % MaskRenderer.palette.count].1) / 255,
                                                       blue: CGFloat(MaskRenderer.palette[f.id % MaskRenderer.palette.count].2) / 255, alpha: 1))
                    if showMasks, let m = f.mask {
                        ctx.draw(Image(decorative: m, scale: 1), in: CGRect(x: ox, y: oy, width: w, height: h))
                    }
                    let r = CGRect(x: ox + f.box.minX * w, y: oy + f.box.minY * h, width: f.box.width * w, height: f.box.height * h)
                    if f.visible {
                        ctx.stroke(Path(roundedRect: r, cornerRadius: 6), with: .color(color), lineWidth: 3)
                    } else {
                        ctx.stroke(Path(roundedRect: r, cornerRadius: 6), with: .color(color.opacity(0.5)), style: StrokeStyle(lineWidth: 2, dash: [6, 6]))
                    }
                    ctx.draw(Text("#\(f.id)\(f.visible ? "" : " (occluded)")").font(.headline).foregroundColor(color),
                             at: CGPoint(x: r.minX, y: r.minY - 12), anchor: .leading)
                }
            }
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let camera: CameraController

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.previewLayer.session = camera.session
        v.previewLayer.videoGravity = .resizeAspectFill
        camera.previewLayer = v.previewLayer
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}
