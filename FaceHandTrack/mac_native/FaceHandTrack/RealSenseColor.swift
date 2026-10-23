import Accelerate
import CoreVideo
import Foundation

/// The colour (2-D RGB) stream of an Intel RealSense camera through Intel's SDK (librealsense).
///
/// macOS lists only the D455f's depth sensor as a camera, so AVFoundation cannot show its colour
/// picture. librealsense reads the camera directly over USB instead; on macOS that needs
/// administrator rights, so the app has to be started with sudo ("Run with RealSense.command").
/// The library is loaded at run time (dlopen) from the folder in FHT_LIBREALSENSE or from Homebrew
/// (`brew install librealsense`, 2.59.1 here), so the app also builds and runs without it. (The
/// pip wheel's 2.56.5 crashes on this Mac while opening the D455f's motion sensor, so it is not used.)
final class RealSenseColor {
    // C API (librealsense2/rs.h). Every handle is an opaque pointer; enums are Int32.
    private typealias Err = UnsafeMutablePointer<OpaquePointer?>
    private typealias FApiVersion = @convention(c) (Err?) -> Int32
    private typealias FCreateContext = @convention(c) (Int32, Err?) -> OpaquePointer?
    private typealias FQueryDevices = @convention(c) (OpaquePointer?, Err?) -> OpaquePointer?
    private typealias FDeviceCount = @convention(c) (OpaquePointer?, Err?) -> Int32
    private typealias FCreateDevice = @convention(c) (OpaquePointer?, Int32, Err?) -> OpaquePointer?
    private typealias FDeviceInfo = @convention(c) (OpaquePointer?, Int32, Err?) -> UnsafePointer<CChar>?
    private typealias FDelete = @convention(c) (OpaquePointer?) -> Void
    private typealias FCreatePipeline = @convention(c) (OpaquePointer?, Err?) -> OpaquePointer?
    private typealias FCreateConfig = @convention(c) (Err?) -> OpaquePointer?
    private typealias FEnableStream = @convention(c) (OpaquePointer?, Int32, Int32, Int32, Int32, Int32, Int32, Err?) -> Void
    private typealias FEnableDevice = @convention(c) (OpaquePointer?, UnsafePointer<CChar>?, Err?) -> Void
    private typealias FStart = @convention(c) (OpaquePointer?, OpaquePointer?, Err?) -> OpaquePointer?
    private typealias FWait = @convention(c) (OpaquePointer?, UInt32, Err?) -> OpaquePointer?
    private typealias FStop = @convention(c) (OpaquePointer?, Err?) -> Void
    private typealias FFrameCount = @convention(c) (OpaquePointer?, Err?) -> Int32
    private typealias FExtract = @convention(c) (OpaquePointer?, Int32, Err?) -> OpaquePointer?
    private typealias FFrameData = @convention(c) (OpaquePointer?, Err?) -> UnsafeRawPointer?
    private typealias FFrameInt = @convention(c) (OpaquePointer?, Err?) -> Int32
    private typealias FErrorMessage = @convention(c) (OpaquePointer?) -> UnsafePointer<CChar>?

    private static let streamColor: Int32 = 2  // RS2_STREAM_COLOR
    private static let formatBGRA8: Int32 = 8  // RS2_FORMAT_BGRA8
    private static let infoName: Int32 = 0, infoSerial: Int32 = 1, infoUSB: Int32 = 9  // rs2_camera_info

    private let lib: UnsafeMutableRawPointer
    private let ctx: OpaquePointer
    private var pipe: OpaquePointer?
    private var thread: Thread?
    private let running = NSLock()
    private var isRunning = false
    private var pool: CVPixelBufferPool?
    private var poolSize = (0, 0)

    /// "Intel RealSense D455F (serial ..., USB 3.2)"
    private(set) var deviceDescription = ""
    /// Mirror the picture like a selfie view (set from any thread).
    var mirrored = true
    /// Called on the capture thread with each new BGRA frame.
    var onFrame: ((CVPixelBuffer) -> Void)?

    /// Folders searched for librealsense2*.dylib (its libusb must sit next to it).
    static func libraryCandidates() -> [String] {
        var dirs: [String] = []
        if let d = ProcessInfo.processInfo.environment["FHT_LIBREALSENSE"] { dirs.append(d) }
        dirs += ["/opt/homebrew/lib", "/usr/local/lib"]
        var files: [String] = []
        for d in dirs {
            let names = (try? FileManager.default.contentsOfDirectory(atPath: d)) ?? []
            for n in names.sorted(by: { $0.count < $1.count }) where n.hasPrefix("librealsense2") && n.hasSuffix(".dylib")
                && !n.contains("-gl") {
                files.append((d as NSString).appendingPathComponent(n))
            }
        }
        return files
    }

    /// Loads librealsense and finds a camera. Throws a readable message if either is missing.
    init() throws {
        var handle: UnsafeMutableRawPointer?
        var tried: [String] = []
        for path in Self.libraryCandidates() {
            if let h = dlopen(path, RTLD_NOW | RTLD_LOCAL) {
                handle = h
                print("[realsense] loaded \(path)")
                break
            }
            tried.append(path + ": " + String(cString: dlerror()))
        }
        guard let handle else {
            throw Self.failure("librealsense not found: install it with  brew install librealsense"
                               + (tried.isEmpty ? "" : "\n" + tried.joined(separator: "\n")))
        }
        lib = handle
        let version: FApiVersion = try Self.symbol(handle, "rs2_get_api_version")
        let create: FCreateContext = try Self.symbol(handle, "rs2_create_context")
        var err: OpaquePointer?
        let v = version(&err)
        try Self.check(handle, err, "rs2_get_api_version")
        guard let c = create(v, &err) else {
            try Self.check(handle, err, "rs2_create_context")
            throw Self.failure("no context")
        }
        try Self.check(handle, err, "rs2_create_context")
        ctx = c
        print("[realsense] librealsense \(v / 10000).\((v / 100) % 100).\(v % 100)")
    }

    deinit {
        stop()
        if let del: FDelete = try? Self.symbol(lib, "rs2_delete_context") { del(ctx) }
    }

    private static func failure(_ msg: String) -> NSError {
        NSError(domain: "RealSense", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }

    private static func symbol<T>(_ lib: UnsafeMutableRawPointer, _ name: String) throws -> T {
        guard let s = dlsym(lib, name) else { throw failure("librealsense has no \(name)") }
        return unsafeBitCast(s, to: T.self)
    }

    private func sym<T>(_ name: String) throws -> T { try Self.symbol(lib, name) }

    private static func check(_ lib: UnsafeMutableRawPointer, _ err: OpaquePointer?, _ call: String) throws {
        guard let err else { return }
        let msgF: FErrorMessage? = try? symbol(lib, "rs2_get_error_message")
        let freeF: FDelete? = try? symbol(lib, "rs2_free_error")
        let msg = msgF?(err).map { String(cString: $0) } ?? "unknown error"
        freeF?(err)
        throw failure("\(call): \(msg)")
    }

    private func check(_ err: OpaquePointer?, _ call: String) throws { try Self.check(lib, err, call) }

    /// Serial numbers and names of the connected cameras.
    func devices() throws -> [(serial: String, name: String, usb: String)] {
        let query: FQueryDevices = try sym("rs2_query_devices")
        let count: FDeviceCount = try sym("rs2_get_device_count")
        let make: FCreateDevice = try sym("rs2_create_device")
        let info: FDeviceInfo = try sym("rs2_get_device_info")
        let delList: FDelete = try sym("rs2_delete_device_list")
        let delDev: FDelete = try sym("rs2_delete_device")
        var err: OpaquePointer?
        let list = query(ctx, &err)
        try check(err, "rs2_query_devices")
        defer { delList(list) }
        let n = count(list, &err)
        try check(err, "rs2_get_device_count")
        var out: [(String, String, String)] = []
        for i in 0..<n {
            guard let d = make(list, i, &err) else { try check(err, "rs2_create_device"); continue }
            defer { delDev(d) }
            func get(_ k: Int32) -> String {
                var e: OpaquePointer?
                let s = info(d, k, &e).map { String(cString: $0) } ?? "?"
                if e != nil { try? check(e, "rs2_get_device_info"); return "?" }
                return s
            }
            out.append((get(Self.infoSerial), get(Self.infoName), get(Self.infoUSB)))
        }
        return out
    }

    /// Starts the colour stream (first camera, or the one with `serial`). Tries 1280x720 @ 30 fps,
    /// then smaller modes (USB 2 ports cannot carry 720p colour at 30 fps).
    func start(serial: String? = nil) throws {
        stop()
        let cams = try devices()
        guard let cam = serial.flatMap({ s in cams.first { $0.serial == s } }) ?? cams.first else {
            throw Self.failure("no RealSense camera found: plug it into a USB 3 port, quit realsense-viewer, "
                               + "and start the app with sudo")
        }
        let createPipe: FCreatePipeline = try sym("rs2_create_pipeline")
        let createCfg: FCreateConfig = try sym("rs2_create_config")
        let enable: FEnableStream = try sym("rs2_config_enable_stream")
        let enableDev: FEnableDevice = try sym("rs2_config_enable_device")
        let startF: FStart = try sym("rs2_pipeline_start_with_config")
        let delCfg: FDelete = try sym("rs2_delete_config")
        let delProfile: FDelete = try sym("rs2_delete_pipeline_profile")
        let delPipe: FDelete = try sym("rs2_delete_pipeline")
        var lastError: Error?
        for (w, h, fps) in [(1280, 720, 30), (848, 480, 30), (640, 480, 30), (640, 480, 15)] {
            var err: OpaquePointer?
            guard let p = createPipe(ctx, &err) else { try check(err, "rs2_create_pipeline"); continue }
            let cfg = createCfg(&err)
            try check(err, "rs2_create_config")
            defer { delCfg(cfg) }
            cam.serial.withCString { enableDev(cfg, $0, &err) }
            try check(err, "rs2_config_enable_device")
            enable(cfg, Self.streamColor, -1, Int32(w), Int32(h), Self.formatBGRA8, Int32(fps), &err)
            try check(err, "rs2_config_enable_stream")
            let profile = startF(p, cfg, &err)
            do {
                try check(err, "rs2_pipeline_start_with_config")
            } catch {
                lastError = error
                delPipe(p)
                continue
            }
            delProfile(profile)
            pipe = p
            deviceDescription = "\(cam.name) (serial \(cam.serial), USB \(cam.usb)) colour \(w)x\(h) @ \(fps) fps"
            print("[realsense] streaming " + deviceDescription)
            running.withLock { isRunning = true }
            let t = Thread { [weak self] in self?.loop() }
            t.name = "facehandtrack.realsense"
            t.qualityOfService = .userInitiated
            thread = t
            t.start()
            return
        }
        throw lastError ?? Self.failure("could not start the colour stream")
    }

    func stop() {
        running.withLock { isRunning = false }
        while thread?.isExecuting == true { usleep(5000) }  // the loop wakes at least once a second
        thread = nil
        guard let p = pipe else { return }
        if let stopF: FStop = try? sym("rs2_pipeline_stop") {
            var err: OpaquePointer?
            stopF(p, &err)
            try? check(err, "rs2_pipeline_stop")
        }
        if let del: FDelete = try? sym("rs2_delete_pipeline") { del(p) }
        pipe = nil
    }

    private func loop() {
        guard let wait: FWait = try? sym("rs2_pipeline_wait_for_frames"),
              let count: FFrameCount = try? sym("rs2_embedded_frames_count"),
              let extract: FExtract = try? sym("rs2_extract_frame"),
              let data: FFrameData = try? sym("rs2_get_frame_data"),
              let width: FFrameInt = try? sym("rs2_get_frame_width"),
              let height: FFrameInt = try? sym("rs2_get_frame_height"),
              let stride: FFrameInt = try? sym("rs2_get_frame_stride_in_bytes"),
              let release: FDelete = try? sym("rs2_release_frame") else { return }
        while running.withLock({ isRunning }) {
            var err: OpaquePointer?
            guard let set = wait(pipe, 1000, &err) else {
                try? check(err, "wait_for_frames")  // timeout: try again
                continue
            }
            defer { release(set) }
            let n = count(set, &err)
            for i in 0..<max(n, 0) {
                guard let f = extract(set, i, &err) else { continue }
                defer { release(f) }
                guard let src = data(f, &err) else { continue }
                let w = Int(width(f, &err)), h = Int(height(f, &err)), rowBytes = Int(stride(f, &err))
                if let pb = copy(src, width: w, height: h, rowBytes: rowBytes) { onFrame?(pb) }
            }
            if err != nil { try? check(err, "frame access") }
        }
    }

    /// librealsense frame (BGRA) -> IOSurface-backed pixel buffer, mirrored if asked.
    private func copy(_ src: UnsafeRawPointer, width: Int, height: Int, rowBytes: Int) -> CVPixelBuffer? {
        if pool == nil || poolSize != (width, height) {
            let attrs: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey: width,
                kCVPixelBufferHeightKey: height,
                kCVPixelBufferIOSurfacePropertiesKey: [:],
            ]
            CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &pool)
            poolSize = (width, height)
        }
        var out: CVPixelBuffer?
        guard let pool, CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out) == kCVReturnSuccess, let pb = out else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        var inBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src), height: vImagePixelCount(height),
                                  width: vImagePixelCount(width), rowBytes: rowBytes)
        var outBuf = vImage_Buffer(data: CVPixelBufferGetBaseAddress(pb), height: vImagePixelCount(height),
                                   width: vImagePixelCount(width), rowBytes: CVPixelBufferGetBytesPerRow(pb))
        if mirrored {
            vImageHorizontalReflect_ARGB8888(&inBuf, &outBuf, vImage_Flags(kvImageNoFlags))
        } else {
            vImageCopyBuffer(&inBuf, &outBuf, 4, vImage_Flags(kvImageNoFlags))
        }
        return pb
    }
}
