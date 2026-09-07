// FVMatcher — thin_FingerVein.dll wrapper for finger-vein matching.
// x86, .NET 8, self-contained publish-single-file.
// No physical device required.
//
// Pipeline insight (from SDK source code analysis):
//   Enrollment: raw_grayscale → cv::imencode(JPEG) → ZHY_Encrypt → fv_extract_model → model
//   Matching:  probe_jpeg → fv_match → (internally: ZHY_Decrypt → imdecode → XLFv_GetFeature → XLFv_MatFeature)
//   Key: fv_match / fv_extract_model expect *encrypted* JPEG bytes (not raw grayscale).
//
// Usage:
//   FVMatcher --match   <probe_image> <model_file>     1:1  comparison
//   FVMatcher --match1n <probe_image> <model_dir>       1:N  identification
//   FVMatcher --enroll  <img1> <img2> <img3> <out>     create template
//   FVMatcher --info                                  SDK info
//
// Image formats:
//   • Raw 320x240 grayscale (76800 bytes) — re-encoded to JPEG then encrypted
//   • Decrypted JPEG file (starts FF D8)  — re-encoded to JPEG then encrypted
//   • The tool handles encoding internally using OpenCV cvEncodeImage.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;

internal static class Program
{
    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------
    private const int ImageW = 320, ImageH = 240, ImageBytes = ImageW * ImageH;
    private const int ModelBytes = 3352;
    private const float DefaultThreshold = 0.63f;

    // -------------------------------------------------------------------------
    // Kernel32 (DLL loading + GetProcAddress)
    // -------------------------------------------------------------------------
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr LoadLibraryW(string lpFileName);
    [DllImport("kernel32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr hModule, string lpProcName);
    [DllImport("kernel32.dll")]
    private static extern bool FreeLibrary(IntPtr hModule);

    // -------------------------------------------------------------------------
    // OpenCV 2.4.3 C API  (opencv_core243.dll, opencv_highgui243.dll)
    // -------------------------------------------------------------------------
    // cvLoadImage / cvReleaseImage already tested; add cvEncodeImage.
    [DllImport("opencv_highgui243.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr cvLoadImage(string filename, int iscolor);  // iscolor=0: grayscale

    [DllImport("opencv_core243.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    private static extern void cvReleaseImage(ref IntPtr img);

    // cvEncodeImage(const char* ext, const CvArr* image) -> IplImage*
    // cvarr is either CvMat* or IplImage* (both start with a pointer).
    [DllImport("opencv_core243.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr cvEncodeImage(string ext, IntPtr image);

    // cvCreateImage(ref IntPtr img, CvSize, nChannels, depth)
    [DllImport("opencv_core243.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    private static extern void cvCreateImage(ref IntPtr img, int width, int height, int nChannels, int depth);

    // cvSetData(IntPtr arr, IntPtr data, int step)
    [DllImport("opencv_core243.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    private static extern void cvSetData(IntPtr arr, IntPtr data, int step);

    // cvGetRow / cvGetCols helpers (return IplImage header view)
    // Actually, use simple approach: cvCreateImage + cvSetData to build IplImage in memory.
    // cvGetImage(const CvArr* arr, IplImage* header) -> IplImage*
    [DllImport("opencv_core243.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr cvGetImage(IntPtr arr, IntPtr header);

    // CvMat is not needed for our use case (we use IplImage directly).

    // OpenCV IplImage x86 struct is 112 bytes.
    // We read fields at known offsets to avoid struct layout mismatches.
    // Field offsets (from OpenCV 2.4.3 types_c.h source):
    //   width=40, height=44, imageSize=64, imageData=68, widthStep=72
    //   imageDataOrigin=100 (estimated; 16 bytes after BorderConst[4] at 84)
    private const int IplOffWidth = 40, IplOffHeight = 44;
    private const int IplOffImageSize = 64, IplOffImageData = 68;
    private const int IplOffWidthStep = 72, IplOffImageDataOrigin = 100;
    private const int IplImageSize = 112;

    private static int ReadInt32(IntPtr base, int offset) =>
        Marshal.ReadInt32(IntPtr.Add(base, offset));

    private static IntPtr ReadIntPtr(IntPtr base, int offset)
    {
        // On x86, IntPtr is 4 bytes = int. On x64, we need to read 4 bytes only.
        int val = Marshal.ReadInt32(IntPtr.Add(base, offset));
        return new IntPtr(val);
    }

    // Helper: encode raw 320x240 grayscale bytes to JPEG using OpenCV.
    // Returns JPEG bytes (without encryption).
    private static byte[] EncodeJpeg(byte[] rawGray)
    {
        if (rawGray.Length != ImageBytes)
            throw new InvalidOperationException($"EncodeJpeg: expected {ImageBytes} bytes, got {rawGray.Length}");

        // Build IplImage in memory: create header + attach our pixel buffer.
        IntPtr ipl = IntPtr.Zero;
        IntPtr headerBuf = Marshal.AllocHGlobal(128); // enough for IplImageHeader
        try
        {
            // Create an IplImage header (not allocated), then set data pointer.
            // cvCreateImage allocates the data buffer; we use a different approach:
            // Create a CvMat, fill it, then cvEncodeImage(cvarr).
            // Actually let's use: create IplImage, fill pixels, encode.
            // cvCreateImage allocates data buffer; we then OVERRIDE imageData.
            cvCreateImage(ref ipl, ImageW, ImageH, 1, 8);  // 1 channel, depth=8
            if (ipl == IntPtr.Zero) throw new InvalidOperationException("cvCreateImage failed");

            var hdr = Marshal.PtrToStructure<IplImageHeader>(ipl);
            if (hdr.imageData == IntPtr.Zero)
                throw new InvalidOperationException("cvCreateImage: imageData is NULL");

            // Copy our grayscale pixels into the OpenCV buffer.
            // Account for widthStep (may be padded).
            for (int r = 0; r < ImageH; r++)
            {
                IntPtr rowDst = IntPtr.Add(hdr.imageData, r * hdr.widthStep);
                int srcOff = r * ImageW;
                if (hdr.widthStep == ImageW)
                    Marshal.Copy(rawGray, srcOff, rowDst, ImageW);
                else
                    for (int c = 0; c < ImageW; c++)
                        Marshal.WriteByte(rowDst, c, rawGray[srcOff + c]);
            }

            // Encode to JPEG.
            IntPtr encIpl = cvEncodeImage(".jpg", ipl);
            if (encIpl == IntPtr.Zero) throw new InvalidOperationException("cvEncodeImage returned NULL");

            var encHdr = Marshal.PtrToStructure<IplImageHeader>(encIpl);
            if (encHdr.imageData == IntPtr.Zero || encHdr.imageSize <= 0)
                throw new InvalidOperationException("cvEncodeImage: imageData is NULL or size=0");

            byte[] jpeg = new byte[encHdr.imageSize];
            Marshal.Copy(encHdr.imageData, jpeg, 0, encHdr.imageSize);
            Console.Error.WriteLine($"  [opencv] encoded raw -> JPEG {jpeg.Length} bytes");
            return jpeg;
        }
        finally
        {
            if (ipl != IntPtr.Zero) cvReleaseImage(ref ipl);
        }
    }

    // -------------------------------------------------------------------------
    // thin_FingerVein.dll  SDK delegates
    // -------------------------------------------------------------------------
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate bool FvMatchDelegate(IntPtr encryptedJpeg, int len, IntPtr model, IntPtr score);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate bool FvExtractModelDelegate(
        IntPtr enc1, int len1, IntPtr enc2, int len2, IntPtr enc3, int len3,
        IntPtr model, IntPtr modelLen);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate bool FvEncryptDelegate(IntPtr src, int len, IntPtr dst);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate bool FvGetDefaultThresholdDelegate(IntPtr score);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void FvLoadFv3DevDelegate();

    // -------------------------------------------------------------------------
    // SDK context
    // -------------------------------------------------------------------------
    private sealed class Sdk : IDisposable
    {
        public IntPtr Handle;
        public FvMatchDelegate? Match;
        public FvExtractModelDelegate? ExtractModel;
        public FvEncryptDelegate? Encrypt;
        public FvGetDefaultThresholdDelegate? GetDefaultThreshold;
        public string DllPath = "";
        public float Threshold = DefaultThreshold;

        private static T? GetFunc<T>(IntPtr h, string name) where T : Delegate
        {
            IntPtr p = GetProcAddress(h, name);
            return p == IntPtr.Zero ? default : (T)Marshal.GetDelegateForFunctionPointer(p, typeof(T));
        }

        public static Sdk Load(string dllPath)
        {
            if (!File.Exists(dllPath)) throw new FileNotFoundException($"SDK not found: {dllPath}");
            var sdk = new Sdk { DllPath = dllPath };
            sdk.Handle = LoadLibraryW(dllPath);
            if (sdk.Handle == IntPtr.Zero)
                throw new InvalidOperationException($"LoadLibraryW failed, 0x{Marshal.GetLastWin32Error():X}");

            GetFunc<FvLoadFv3DevDelegate>(sdk.Handle, "fv_Load_FingerVein_v3_dev")?.Invoke();
            sdk.Match = GetFunc<FvMatchDelegate>(sdk.Handle, "fv_match");
            sdk.ExtractModel = GetFunc<FvExtractModelDelegate>(sdk.Handle, "fv_extract_model");
            sdk.Encrypt = GetFunc<FvEncryptDelegate>(sdk.Handle, "fv_Encrypt");
            sdk.GetDefaultThreshold = GetFunc<FvGetDefaultThresholdDelegate>(sdk.Handle, "fv_get_default_threshold");

            if (sdk.Match == null) throw new InvalidOperationException("fv_match not found");
            if (sdk.ExtractModel == null) throw new InvalidOperationException("fv_extract_model not found");
            if (sdk.Encrypt == null) throw new InvalidOperationException("fv_Encrypt not found");

            IntPtr p = Marshal.AllocHGlobal(4);
            if (sdk.GetDefaultThreshold != null && sdk.GetDefaultThreshold(p))
            {
                float t = BitConverter.ToSingle(BitConverter.GetBytes(Marshal.ReadInt32(p)), 0);
                if (!float.IsNaN(t) && !float.IsInfinity(t) && t > 0 && t < 10) sdk.Threshold = t;
            }
            Marshal.FreeHGlobal(p);

            Console.Error.WriteLine($"  [sdk] DLL={Path.GetFileName(dllPath)}, threshold={sdk.Threshold}");
            return sdk;
        }

        public void Dispose() { if (Handle != IntPtr.Zero) { FreeLibrary(Handle); Handle = IntPtr.Zero; } }
    }

    // -------------------------------------------------------------------------
    // Image pipeline: file → raw gray → JPEG → encrypted bytes
    // -------------------------------------------------------------------------

    /// <summary>
    /// Load image file and return raw 320x240 grayscale bytes (76800).
    /// Decrypts or decodes as needed.
    /// </summary>
    private static byte[] LoadImageAsRaw(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"Image not found: {path}");
        byte[] raw = File.ReadAllBytes(path);

        if (raw.Length >= 2 && raw[0] == 0xFF && raw[1] == 0xD8)
        {
            // JPEG: decode via cvLoadImage to raw grayscale.
            string tmp = Path.Combine(Path.GetTempPath(), $"fv_j_{Guid.NewGuid():N}.jpg");
            try
            {
                File.WriteAllBytes(tmp, raw);
                IntPtr ipl = cvLoadImage(tmp, 0);  // 0 = grayscale
                if (ipl == IntPtr.Zero) throw new InvalidOperationException($"cvLoadImage failed: {path}");
                try
                {
                    var hdr = Marshal.PtrToStructure<IplImageHeader>(ipl);
                    if (hdr.imageData == IntPtr.Zero)
                        throw new InvalidOperationException("cvLoadImage: imageData is NULL");
                    if (hdr.width != ImageW || hdr.height != ImageH)
                        throw new InvalidOperationException($"cvLoadImage: {hdr.width}x{hdr.height}, expected {ImageW}x{ImageH}");

                    byte[] pixels = new byte[ImageBytes];
                    for (int r = 0; r < ImageH; r++)
                    {
                        IntPtr rowPtr = IntPtr.Add(hdr.imageData, r * hdr.widthStep);
                        Marshal.Copy(rowPtr, pixels, r * ImageW, ImageW);
                    }
                    Console.Error.WriteLine($"  [jpeg] cvLoadImage -> {pixels.Length} raw bytes");
                    return pixels;
                }
                finally { cvReleaseImage(ref ipl); }
            }
            finally { try { File.Delete(tmp); } catch { } }
        }

        if (raw.Length == ImageBytes)
        {
            Console.Error.WriteLine($"  [raw] {raw.Length} bytes as-is");
            return raw;
        }

        throw new InvalidOperationException($"Image {path} is neither JPEG nor {ImageBytes} raw bytes (size={raw.Length}).");
    }

    // -------------------------------------------------------------------------
    // SDK call helpers  (encrypt-then-match / encrypt-then-extract)
    // -------------------------------------------------------------------------

    /// <summary>
    /// Encrypt bytes with fv_Encrypt (16-byte block cipher from thin_FingerVein.dll).
    /// Output length = 16-byte aligned.
    /// </summary>
    private static byte[] EncryptBytes(Sdk sdk, byte[] src)
    {
        int aligned = (src.Length + 15) / 16 * 16;
        IntPtr pSrc = Marshal.AllocHGlobal(aligned);
        IntPtr pDst = Marshal.AllocHGlobal(aligned);
        try
        {
            Marshal.Copy(src, 0, pSrc, src.Length);
            for (int i = src.Length; i < aligned; i++) Marshal.WriteByte(pSrc, i, 0);
            bool ok = sdk.Encrypt!(pSrc, src.Length, pDst);
            if (!ok) throw new InvalidOperationException("fv_Encrypt returned FALSE");
            byte[] dst = new byte[aligned];
            Marshal.Copy(pDst, dst, 0, aligned);
            return dst;
        }
        finally { Marshal.FreeHGlobal(pSrc); Marshal.FreeHGlobal(pDst); }
    }

    private static (bool ok, float score) CallMatch(Sdk sdk, byte[] rawGray, byte[] model)
    {
        if (model.Length != ModelBytes)
            throw new InvalidOperationException($"model must be {ModelBytes} bytes, got {model.Length}");

        // 1. Encode to JPEG
        byte[] jpeg = EncodeJpeg(rawGray);

        // 2. Encrypt
        byte[] enc = EncryptBytes(sdk, jpeg);

        // 3. Call SDK
        IntPtr pEnc = IntPtr.Zero, pModel = IntPtr.Zero, pScore = IntPtr.Zero;
        try
        {
            pEnc = Marshal.AllocHGlobal(enc.Length);
            pModel = Marshal.AllocHGlobal(model.Length);
            pScore = Marshal.AllocHGlobal(4);
            Marshal.Copy(enc, 0, pEnc, enc.Length);
            Marshal.Copy(model, 0, pModel, model.Length);
            Marshal.WriteInt32(pScore, 0, 0);

            bool ok = sdk.Match!(pEnc, enc.Length, pModel, pScore);
            float score = BitConverter.ToSingle(BitConverter.GetBytes(Marshal.ReadInt32(pScore)), 0);
            Console.Error.WriteLine($"  [sdk] fv_match -> ok={ok}, score={score:F6}");
            return (ok, score);
        }
        finally { if (pEnc != IntPtr.Zero) Marshal.FreeHGlobal(pEnc); if (pModel != IntPtr.Zero) Marshal.FreeHGlobal(pModel); if (pScore != IntPtr.Zero) Marshal.FreeHGlobal(pScore); }
    }

    private static (bool ok, byte[] model) CallExtractModel(Sdk sdk, byte[] raw1, byte[] raw2, byte[] raw3)
    {
        byte[] j1 = EncodeJpeg(raw1);
        byte[] j2 = EncodeJpeg(raw2);
        byte[] j3 = EncodeJpeg(raw3);
        byte[] e1 = EncryptBytes(sdk, j1);
        byte[] e2 = EncryptBytes(sdk, j2);
        byte[] e3 = EncryptBytes(sdk, j3);

        IntPtr p1 = IntPtr.Zero, p2 = IntPtr.Zero, p3 = IntPtr.Zero;
        IntPtr pModel = IntPtr.Zero, pLen = IntPtr.Zero;
        try
        {
            p1 = Marshal.AllocHGlobal(e1.Length); p2 = Marshal.AllocHGlobal(e2.Length); p3 = Marshal.AllocHGlobal(e3.Length);
            pModel = Marshal.AllocHGlobal(ModelBytes); pLen = Marshal.AllocHGlobal(4);
            Marshal.Copy(e1, 0, p1, e1.Length); Marshal.Copy(e2, 0, p2, e2.Length); Marshal.Copy(e3, 0, p3, e3.Length);
            Marshal.WriteInt32(pLen, 0, ModelBytes);

            bool ok = sdk.ExtractModel!(p1, e1.Length, p2, e2.Length, p3, e3.Length, pModel, pLen);
            int written = Marshal.ReadInt32(pLen);
            if (written <= 0) written = ModelBytes;
            byte[] model = new byte[written];
            Marshal.Copy(pModel, model, 0, written);
            Console.Error.WriteLine($"  [sdk] fv_extract_model -> ok={ok}, model={written} bytes");
            return (ok, model);
        }
        finally { foreach (var p in new[] { p1, p2, p3, pModel, pLen }) if (p != IntPtr.Zero) Marshal.FreeHGlobal(p); }
    }

    // -------------------------------------------------------------------------
    // JSON helpers
    // -------------------------------------------------------------------------
    private static void WriteJson(object o) =>
        Console.WriteLine(JsonSerializer.Serialize(o, new JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));

    private static void Error(string code, string msg)
    { WriteJson(new { ok = false, error = new { code, message = msg } }); Environment.ExitCode = 1; }

    // -------------------------------------------------------------------------
    // Commands
    // -------------------------------------------------------------------------
    private static int CmdMatch(Sdk sdk, string probePath, string modelPath)
    {
        byte[] raw = LoadImageAsRaw(probePath);
        byte[] model = File.ReadAllBytes(modelPath);
        if (model.Length != ModelBytes) { Error("model_length", $"expected {ModelBytes}, got {model.Length}"); return 1; }
        var (ok, score) = CallMatch(sdk, raw, model);
        WriteJson(new { ok, operation = "match", score, threshold = sdk.Threshold, isMatch = ok && score >= sdk.Threshold });
        return ok ? 0 : 2;
    }

    private static int CmdMatch1N(Sdk sdk, string probePath, string modelDir)
    {
        if (!Directory.Exists(modelDir)) { Error("dir_missing", modelDir); return 1; }
        byte[] raw = LoadImageAsRaw(probePath);
        var jpeg = EncodeJpeg(raw);
        var enc = EncryptBytes(sdk, jpeg);

        var modelFiles = Directory.EnumerateFiles(modelDir, "*.*", SearchOption.AllDirectories)
            .Where(f => { try { return new FileInfo(f).Length == ModelBytes; } catch { return false; } })
            .OrderBy(f => f).ToList();

        if (modelFiles.Count == 0) { Error("no_models", $"no {ModelBytes}-byte models in {modelDir}"); return 1; }

        IntPtr pEnc = IntPtr.Zero, pModel = IntPtr.Zero, pScore = IntPtr.Zero;
        pEnc = Marshal.AllocHGlobal(enc.Length);
        pModel = Marshal.AllocHGlobal(ModelBytes);
        pScore = Marshal.AllocHGlobal(4);
        Marshal.Copy(enc, 0, pEnc, enc.Length);

        var results = new List<object>();
        float best = float.MinValue; string? bestPath = null;
        int errors = 0;
        for (int i = 0; i < modelFiles.Count; i++)
        {
            string fp = modelFiles[i];
            try
            {
                byte[] m = File.ReadAllBytes(fp);
                if (m.Length != ModelBytes) { errors++; continue; }
                Marshal.Copy(m, 0, pModel, ModelBytes);
                Marshal.WriteInt32(pScore, 0, 0);
                bool ok = sdk.Match!(pEnc, enc.Length, pModel, pScore);
                float score = BitConverter.ToSingle(BitConverter.GetBytes(Marshal.ReadInt32(pScore)), 0);
                bool hit = score >= sdk.Threshold;
                results.Add(new { path = fp, index = i, ok = true, score, hit });
                if (score > best) { best = score; bestPath = fp; }
            }
            catch (Exception ex) { results.Add(new { path = fp, index = i, ok = false, score = 0.0, hit = false, error = ex.Message }); errors++; }
        }
        Marshal.FreeHGlobal(pEnc); Marshal.FreeHGlobal(pModel); Marshal.FreeHGlobal(pScore);

        var ranked = results.OrderByDescending(r => (float)typeof(object).GetField("score")!.GetValue(r)).ToList();
        WriteJson(new
        {
            ok = errors < modelFiles.Count,
            operation = "match1n",
            modelDir,
            threshold = sdk.Threshold,
            total = modelFiles.Count,
            errors,
            best = new { path = bestPath, score = best, hit = best >= sdk.Threshold },
            results = ranked,
        });
        return best >= sdk.Threshold ? 0 : 2;
    }

    private static int CmdEnroll(Sdk sdk, string p1, string p2, string p3, string outPath)
    {
        var r1 = LoadImageAsRaw(p1); var r2 = LoadImageAsRaw(p2); var r3 = LoadImageAsRaw(p3);
        var (ok, model) = CallExtractModel(sdk, r1, r2, r3);
        if (!ok) { Error("extract_failed", "fv_extract_model returned FALSE"); return 2; }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllBytes(outPath, model);
        WriteJson(new { ok = true, operation = "enroll", images = new[] { p1, p2, p3 }, model = new { path = outPath, bytes = model.Length } });
        return 0;
    }

    private static int CmdInfo(Sdk sdk) { WriteJson(new { ok = true, sdk = new { dll = sdk.DllPath, threshold = sdk.Threshold, modelBytes = ModelBytes, imageSize = $"{ImageW}x{ImageH}" } }); return 0; }

    private static void PrintUsage()
    {
        Console.Error.WriteLine("FVMatcher — thin_FingerVein.dll wrapper (no device needed)");
        Console.Error.WriteLine("Commands:");
        Console.Error.WriteLine("  --match   <probe> <model_file>     1:1 match");
        Console.Error.WriteLine("  --match1n <probe> <model_dir>      1:N identification");
        Console.Error.WriteLine("  --enroll  <img1> <img2> <img3> <out.dat>  create template");
        Console.Error.WriteLine("  --info                                SDK info");
        Console.Error.WriteLine("Image: raw 320x240 / decrypted JPEG — encoding handled automatically.");
    }

    // -------------------------------------------------------------------------
    // Entry point
    // -------------------------------------------------------------------------
    private static int Main(string[] args)
    {
        try
        {
            // Load OpenCV (cvLoadImage, cvEncodeImage, cvReleaseImage).
            string opencvPath = @"D:\WorkSpace\Research\Research\ENG-SDK-C++\DEMO\Release\opencv_highgui243.dll";
            IntPtr ocv = LoadLibraryW(opencvPath);
            if (ocv == IntPtr.Zero) { Console.Error.WriteLine($"WARNING: OpenCV not loaded: {opencvPath}"); }
            else { Console.Error.WriteLine($"  [opencv] {opencvPath}"); FreeLibrary(ocv); }

            // Load SDK.
            string sdkPath = new[] {
                @"D:\WorkSpace\Research\Research\ENG-SDK-C++\DEMO\Release\thin_FingerVein.dll",
                @"D:\WorkSpace\Research\Research\ENG-SDK-C++\DEMO\Release\FingerVein_v3_dev.dll",
            }.FirstOrDefault(File.Exists) ?? throw new FileNotFoundException("thin_FingerVein.dll not found");

            Console.Error.WriteLine($"SDK: {sdkPath}");
            using Sdk sdk = Sdk.Load(sdkPath);

            if (args.Length == 0) { PrintUsage(); return 1; }
            string cmd = args[0];
            int rc = cmd switch
            {
                "--match" when args.Length == 3 => CmdMatch(sdk, args[1], args[2]),
                "--match1n" when args.Length == 3 => CmdMatch1N(sdk, args[1], args[2]),
                "--enroll" when args.Length == 5 => CmdEnroll(sdk, args[1], args[2], args[3], args[4]),
                "--info" => CmdInfo(sdk),
                "-h" or "--help" => (PrintUsage(), 0).Item2,
                _ => (Error("unknown", $"unknown command: {cmd}"), 1).Item2,
            };
            return rc;
        }
        catch (Exception ex) { Error("exception", $"{ex.GetType().Name}: {ex.Message}"); return 1; }
    }
}
