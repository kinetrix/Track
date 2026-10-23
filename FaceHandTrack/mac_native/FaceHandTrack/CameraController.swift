import AVFoundation
import CoreVideo

/// macOS camera: 1280x720 BGRA frames from any camera AVFoundation lists (built-in,
/// USB/UVC such as the RealSense colour sensor, iPhone Continuity), optionally mirrored.
/// Frames and preview are mirrored the same way, so the overlay lines up.
final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "facehandtrack.camera", qos: .userInitiated)
    private(set) var device: AVCaptureDevice?
    private(set) var mirrored = true
    /// Set by CameraPreview so the preview is mirrored like the frames.
    weak var previewLayer: AVCaptureVideoPreviewLayer?
    var onFrame: ((CVPixelBuffer) -> Void)?

    /// A RealSense's colour sensor ("... RGB"). macOS may list only the depth sensor ("... Depth"),
    /// which gives no usable picture, so that one is never chosen automatically.
    static func isRealSenseColor(_ d: AVCaptureDevice) -> Bool {
        let n = d.localizedName.lowercased()
        return n.contains("realsense") && (n.contains("rgb") || n.contains("color") || n.contains("colour"))
    }

    /// Every video camera, in the order the app prefers: RealSense colour sensor, the Mac's own
    /// camera, other cameras, and last any RealSense depth / infrared sensor.
    static func cameras() -> [AVCaptureDevice] {
        let types: [AVCaptureDevice.DeviceType] = [.builtInWideAngleCamera, .external, .continuityCamera]
        let found = AVCaptureDevice.DiscoverySession(deviceTypes: types, mediaType: .video, position: .unspecified).devices
        func rank(_ d: AVCaptureDevice) -> Int {
            if isRealSenseColor(d) { return 0 }
            if d.localizedName.lowercased().contains("realsense") { return 4 }
            if d.deviceType == .builtInWideAngleCamera { return 1 }
            return 2
        }
        return found.sorted { (rank($0), $0.localizedName) < (rank($1), $1.localizedName) }
    }

    func start(device: AVCaptureDevice?, mirrored: Bool) {
        queue.async {
            self.mirrored = mirrored
            self.configure(device ?? Self.cameras().first)
            if !self.session.isRunning { self.session.startRunning() }
        }
    }

    func select(_ device: AVCaptureDevice) {
        queue.async { self.configure(device) }
    }

    func setMirrored(_ on: Bool) {
        queue.async {
            self.mirrored = on
            self.applyMirroring()
        }
    }

    func stop() { queue.async { self.session.stopRunning() } }

    private func configure(_ device: AVCaptureDevice?) {
        guard let device else {
            print("[camera] no camera found")
            return
        }
        session.beginConfiguration()
        defer {
            session.commitConfiguration()
            applyMirroring()
        }
        session.inputs.forEach { session.removeInput($0) }
        // The detector sees 640x384 and EdgeTAM 512x512: more pixels buy nothing.
        session.sessionPreset = .high
        guard let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else {
            print("[camera] cannot open \(device.localizedName)")
            return
        }
        session.addInput(input)
        if session.canSetSessionPreset(.hd1280x720) { session.sessionPreset = .hd1280x720 }
        if session.outputs.isEmpty {
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: queue)
            if session.canAddOutput(output) { session.addOutput(output) }
        }
        self.device = device
        let d = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        print("[camera] using \(device.localizedName) (\(d.width)x\(d.height))")
    }

    private func applyMirroring() {
        for c in [output.connection(with: .video), previewLayer?.connection].compactMap({ $0 }) where c.isVideoMirroringSupported {
            c.automaticallyAdjustsVideoMirroring = false
            c.isVideoMirrored = mirrored
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(pb)
    }
}
