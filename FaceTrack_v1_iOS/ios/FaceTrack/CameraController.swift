import AVFoundation
import UIKit

/// 720p BGRA frames, rotated upright and mirrored for the front camera, so the
/// buffers the models see match what the preview layer shows. Works on iPhone
/// and on Apple-silicon Macs ("Designed for iPhone"), where the FaceTime camera
/// is landscape and needs no rotation.
final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "facetrack.camera", qos: .userInitiated)
    private(set) var position: AVCaptureDevice.Position = .front
    private var rotation: AVCaptureDevice.RotationCoordinator?
    /// Set by CameraPreview so rotation/mirroring of preview and frames stay identical.
    weak var previewLayer: AVCaptureVideoPreviewLayer?
    var onFrame: ((CVPixelBuffer) -> Void)?

    func start() {
        queue.async {
            self.configure(position: self.position)
            self.session.startRunning()
        }
    }

    func stop() { queue.async { self.session.stopRunning() } }

    func flip() {
        queue.async {
            self.position = self.position == .front ? .back : .front
            self.configure(position: self.position)
        }
    }

    private func configure(position: AVCaptureDevice.Position) {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        // Detector sees 360x640 and EdgeTAM 512x512: more pixels buy nothing.
        session.sessionPreset = session.canSetSessionPreset(.hd1280x720) ? .hd1280x720 : .high
        session.inputs.forEach { session.removeInput($0) }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
            ?? AVCaptureDevice.default(for: .video),  // Mac: FaceTime / Continuity camera
            let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else { return }
        session.addInput(input)
        if session.outputs.isEmpty {
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: queue)
            if session.canAddOutput(output) { session.addOutput(output) }
        }

        // Upright frames for whatever camera this is (90° on iPhone portrait, 0° on a Mac).
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
        rotation = coordinator
        let mirror = device.position != .back
        // On a Mac the camera is already upright; the coordinator assumes a portrait iPhone.
        let onMac = ProcessInfo.processInfo.isiOSAppOnMac
        if let c = output.connection(with: .video) {
            let angle = onMac ? 0 : coordinator.videoRotationAngleForHorizonLevelCapture
            if c.isVideoRotationAngleSupported(angle) { c.videoRotationAngle = angle }
            if c.isVideoMirroringSupported {
                c.automaticallyAdjustsVideoMirroring = false
                c.isVideoMirrored = mirror
            }
        }
        if let pc = previewLayer?.connection {
            let angle = onMac ? 0 : coordinator.videoRotationAngleForHorizonLevelPreview
            if pc.isVideoRotationAngleSupported(angle) { pc.videoRotationAngle = angle }
            if pc.isVideoMirroringSupported {
                pc.automaticallyAdjustsVideoMirroring = false
                pc.isVideoMirrored = mirror
            }
        }

        // Prefer 60 fps. The pipeline always takes the newest frame, so after finishing one it
        // waits for the next to arrive: at 30 fps a 37 ms frame means waiting until 66 ms
        // (15 FPS); at 60 fps the next frame is there by 50 ms (20 FPS).
        selectFrameRate(device, preferred: 60)
    }

    /// If the current format can't reach `preferred` fps, switch to the smallest 8-bit format
    /// of at least 1280x720 that can; then allow up to `preferred` fps (or the format's max).
    /// The camera may still drop to a lower rate in low light.
    private func selectFrameRate(_ device: AVCaptureDevice, preferred: Double) {
        func dims(_ f: AVCaptureDevice.Format) -> CMVideoDimensions { CMVideoFormatDescriptionGetDimensions(f.formatDescription) }
        func maxRate(_ f: AVCaptureDevice.Format) -> Double { f.videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0 }
        let eightBit: Set<FourCharCode> = [kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        guard (try? device.lockForConfiguration()) != nil else { return }
        defer { device.unlockForConfiguration() }
        if maxRate(device.activeFormat) < preferred {
            let candidates = device.formats.filter { f in
                let d = dims(f)
                return d.width >= 1280 && d.height >= 720 && d.width <= 1920 && maxRate(f) >= preferred
                    && eightBit.contains(CMFormatDescriptionGetMediaSubType(f.formatDescription))
            }
            if let best = candidates.min(by: { Int(dims($0).width) * Int(dims($0).height) < Int(dims($1).width) * Int(dims($1).height) }) {
                device.activeFormat = best  // the session switches to .inputPriority
            }
        }
        let fps = Int32(min(preferred, maxRate(device.activeFormat)).rounded(.down))
        if fps > 0 { device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: fps) }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(pb)
    }
}
