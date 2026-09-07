// FVVein — Clean C# finger-vein enrollment + matching tool
// No IplImage, no OpenCV P/Invoke — pure managed .NET only.
// Uses the thin_FingerVein.dll SDK for enrollment/matching logic.
//
// Pipeline (SDK-native):
//   Enrollment: 3 raw-grayscale images → fv_extract_model → 3352-byte model file
//   Matching:   probe image → fv_match → similarity score
//
// Usage:
//   FVVein enroll   <img1> <img2> <img3> <output_model.dat>
//   FVVein match    <probe_image> <model_file>
//   FVVein match1n  <probe_image> <models_dir>
//   FVVein info

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;

internal static class Program
{
    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------
    private const int ImageW = 320, ImageH = 240, ImageBytes = ImageW * ImageH;
    private const int ModelBytes = 3352;
    private const float DefaultThreshold = 0.63f;

    // -----------------------------------------------------------------------
    // Kernel32 (LoadLibrary + GetProcAddress)
    // -----------------------------------------------------------------------
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr LoadLibraryW(string lpFileName);

    [DllImport("kernel32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr hModule, string lpProcName);

    [DllImport("kernel32.dll")]
    private static extern bool FreeLibrary(IntPtr hModule);

    // -----------------------------------------------------------------------
    // SDK delegates (StdCall, x86 thin_FingerVein.dll)
    // Use IntPtr for byte buffers — explicit Marshal.AllocHGlobal/Copy/Read.
    // This is the only reliable way to interop with raw C buffers.
    // -----------------------------------------------------------------------
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate bool FvMatchDelegate(
        IntPtr encryptedJpeg, int jpegLen,
        IntPtr model, int modelLen,
        out float score
    );

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate bool FvExtractModelDelegate(
        IntPtr enc1, int len1,
        IntPtr enc2, int len2,
        IntPtr enc3, int len3,
        IntPtr model, ref int modelLen
    );

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate bool FvEncryptDelegate(
        IntPtr src, int srcLen,
        IntPtr dst
    );

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate bool FvGetDefaultThresholdDelegate(out float threshold);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate void FvLoadFv3DevDelegate();

    // -----------------------------------------------------------------------
    // SDK context — clean managed wrapper, no OpenCV structs
    // -----------------------------------------------------------------------
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
            if (!File.Exists(dllPath))
                throw new FileNotFoundException($"SDK not found: {dllPath}");

            var sdk = new Sdk { DllPath = dllPath };
            sdk.Handle = LoadLibraryW(dllPath);
            if (sdk.Handle == IntPtr.Zero)
                throw new InvalidOperationException(
                    $"LoadLibraryW failed for {dllPath}, error 0x{Marshal.GetLastWin32Error():X}");

            // Initialize
            GetFunc<FvLoadFv3DevDelegate>(sdk.Handle, "fv_Load_FingerVein_v3_dev")?.Invoke();

            sdk.Match = GetFunc<FvMatchDelegate>(sdk.Handle, "fv_match");
            sdk.ExtractModel = GetFunc<FvExtractModelDelegate>(sdk.Handle, "fv_extract_model");
            sdk.Encrypt = GetFunc<FvEncryptDelegate>(sdk.Handle, "fv_Encrypt");
            sdk.GetDefaultThreshold = GetFunc<FvGetDefaultThresholdDelegate>(sdk.Handle, "fv_get_default_threshold");

            if (sdk.Match == null)
                throw new InvalidOperationException("fv_match not found in DLL");
            if (sdk.ExtractModel == null)
                throw new InvalidOperationException("fv_extract_model not found in DLL");
            if (sdk.Encrypt == null)
                throw new InvalidOperationException("fv_Encrypt not found in DLL");

            // Query default threshold
            if (sdk.GetDefaultThreshold != null && sdk.GetDefaultThreshold(out float t))
            {
                if (!float.IsNaN(t) && !float.IsInfinity(t) && t > 0 && t < 10)
                    sdk.Threshold = t;
            }

            Console.Error.WriteLine($"[sdk] DLL={Path.GetFileName(dllPath)}, threshold={sdk.Threshold}");
            return sdk;
        }

        public void Dispose()
        {
            if (Handle != IntPtr.Zero)
            {
                FreeLibrary(Handle);
                Handle = IntPtr.Zero;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Image loading — managed .NET only (no IplImage, no OpenCV)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Load a finger-vein image file and return raw 320x240 grayscale bytes (76800).
    /// Supported formats: raw 76800-byte binary, JPEG (FF D8 header).
    /// Uses System.Drawing for JPEG decoding — no OpenCV needed.
    /// </summary>
    private static byte[] LoadImageRaw(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Image not found: {path}");

        byte[] raw = File.ReadAllBytes(path);

        // JPEG — decode with System.Drawing
        if (raw.Length >= 2 && raw[0] == 0xFF && raw[1] == 0xD8)
        {
            using var ms = new MemoryStream(raw);
            using var bmp = new System.Drawing.Bitmap(ms);
            if (bmp.Width != ImageW || bmp.Height != ImageH)
                throw new InvalidOperationException(
                    $"JPEG is {bmp.Width}x{bmp.Height}, expected {ImageW}x{ImageH}");

            byte[] pixels = new byte[ImageBytes];
            for (int r = 0; r < ImageH; r++)
            {
                for (int c = 0; c < ImageW; c++)
                {
                    var p = bmp.GetPixel(c, r);
                    // Grayscale = luminance
                    int gray = (int)(0.299 * p.R + 0.587 * p.G + 0.114 * p.B);
                    pixels[r * ImageW + c] = (byte)Math.Clamp(gray, 0, 255);
                }
            }
            Console.Error.WriteLine($"[jpeg] decoded via System.Drawing -> {pixels.Length} bytes");
            return pixels;
        }

        // Raw grayscale bytes
        if (raw.Length == ImageBytes)
        {
            Console.Error.WriteLine($"[raw] {raw.Length} bytes as-is");
            return raw;
        }

        throw new InvalidOperationException(
            $"Image {path} is neither JPEG (FF D8) nor {ImageBytes}-byte raw grayscale (size={raw.Length}).");
    }

    // -----------------------------------------------------------------------
    // JPEG encoding — pure managed (no cvEncodeImage, no OpenCV)
    // -----------------------------------------------------------------------

    /// <summary>
    /// Encode raw 320x240 grayscale bytes to JPEG using System.Drawing.
    /// Matches the SDK's expected input format: 8bpp grayscale → JPEG.
    /// </summary>
    private static byte[] EncodeJpeg(byte[] rawGray)
    {
        if (rawGray.Length != ImageBytes)
            throw new InvalidOperationException(
                $"EncodeJpeg: expected {ImageBytes} bytes, got {rawGray.Length}");

        // Use 24bpp RGB so the JPEG encoder doesn't try chroma subsampling on indexed pixels
        using var bmp = new System.Drawing.Bitmap(ImageW, ImageH, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
        var rect = new System.Drawing.Rectangle(0, 0, ImageW, ImageH);
        var bmpData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.WriteOnly, bmp.PixelFormat);
        try
        {
            IntPtr rowPtr = bmpData.Scan0;
            int stride = bmpData.Stride;  // 3 * 320 = 960 (with padding)
            for (int r = 0; r < ImageH; r++)
            {
                IntPtr dest = IntPtr.Add(rowPtr, r * stride);
                int srcOff = r * ImageW;
                // Expand grayscale to RGB
                for (int c = 0; c < ImageW; c++)
                {
                    byte g = rawGray[srcOff + c];
                    Marshal.WriteByte(dest, c * 3 + 0, g); // B
                    Marshal.WriteByte(dest, c * 3 + 1, g); // G
                    Marshal.WriteByte(dest, c * 3 + 2, g); // R
                }
            }
        }
        finally
        {
            bmp.UnlockBits(bmpData);
        }

        // Encode to JPEG via MemoryStream with explicit quality
        var jpegCodec = System.Drawing.Imaging.ImageCodecInfo.GetImageEncoders()
            .First(c => c.FormatID == System.Drawing.Imaging.ImageFormat.Jpeg.Guid);
        var encParams = new System.Drawing.Imaging.EncoderParameters(1);
        encParams.Param[0] = new System.Drawing.Imaging.EncoderParameter(
            System.Drawing.Imaging.Encoder.Quality, 90L);

        using var ms = new MemoryStream();
        bmp.Save(ms, jpegCodec, encParams);
        byte[] jpeg = ms.ToArray();
        Console.Error.WriteLine($"[opencv-free] encoded raw -> JPEG {jpeg.Length} bytes");
        return jpeg;
    }

    // -----------------------------------------------------------------------
    // SDK helpers
    // -----------------------------------------------------------------------

    private static byte[] EncryptBytes(Sdk sdk, byte[] src)
    {
        int aligned = (src.Length + 15) / 16 * 16;
        byte[] dst = new byte[aligned];
        IntPtr pSrc = Marshal.AllocHGlobal(aligned);
        IntPtr pDst = Marshal.AllocHGlobal(aligned);
        try
        {
            Marshal.Copy(src, 0, pSrc, src.Length);
            for (int i = src.Length; i < aligned; i++) Marshal.WriteByte(pSrc, i, 0);
            bool ok = sdk.Encrypt!(pSrc, src.Length, pDst);
            if (!ok)
                throw new InvalidOperationException("fv_Encrypt returned FALSE");
            Marshal.Copy(pDst, dst, 0, aligned);
            return dst;
        }
        finally
        {
            Marshal.FreeHGlobal(pSrc);
            Marshal.FreeHGlobal(pDst);
        }
    }

    private static (bool ok, float score) CallMatch(Sdk sdk, byte[] rawGray, byte[] model)
    {
        if (model.Length != ModelBytes)
            throw new InvalidOperationException($"model must be {ModelBytes} bytes, got {model.Length}");

        byte[] jpeg = EncodeJpeg(rawGray);
        byte[] enc = EncryptBytes(sdk, jpeg);

        IntPtr pEnc = Marshal.AllocHGlobal(enc.Length);
        IntPtr pModel = Marshal.AllocHGlobal(model.Length);
        try
        {
            Marshal.Copy(enc, 0, pEnc, enc.Length);
            Marshal.Copy(model, 0, pModel, model.Length);

            bool ok = sdk.Match!(pEnc, enc.Length, pModel, model.Length, out float score);
            Console.Error.WriteLine($"[sdk] fv_match -> ok={ok}, score={score:F6}");
            return (ok, score);
        }
        finally
        {
            Marshal.FreeHGlobal(pEnc);
            Marshal.FreeHGlobal(pModel);
        }
    }

    /// <summary>
    /// Safe wrapper around CallMatch that traps access violations inside the
    /// closed-source SDK and reports them as a structured error instead of
    /// letting the whole process abort.
    /// </summary>
    private static (bool ok, float score) SafeCallMatch(Sdk sdk, byte[] rawGray, byte[] model)
    {
        try
        {
            return CallMatch(sdk, rawGray, model);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[sdk] fv_match crashed: {ex.GetType().Name}: {ex.Message}");
            return (false, 0f);
        }
    }

    private static (bool ok, byte[] model) CallExtractModel(Sdk sdk, byte[] raw1, byte[] raw2, byte[] raw3)
    {
        byte[] j1 = EncodeJpeg(raw1);
        byte[] j2 = EncodeJpeg(raw2);
        byte[] j3 = EncodeJpeg(raw3);
        byte[] e1 = EncryptBytes(sdk, j1);
        byte[] e2 = EncryptBytes(sdk, j2);
        byte[] e3 = EncryptBytes(sdk, j3);

        int modelLen = ModelBytes;
        byte[] model = new byte[ModelBytes];

        IntPtr p1 = Marshal.AllocHGlobal(e1.Length);
        IntPtr p2 = Marshal.AllocHGlobal(e2.Length);
        IntPtr p3 = Marshal.AllocHGlobal(e3.Length);
        IntPtr pModel = Marshal.AllocHGlobal(ModelBytes);
        try
        {
            Marshal.Copy(e1, 0, p1, e1.Length);
            Marshal.Copy(e2, 0, p2, e2.Length);
            Marshal.Copy(e3, 0, p3, e3.Length);

            bool ok = sdk.ExtractModel!(
                p1, e1.Length,
                p2, e2.Length,
                p3, e3.Length,
                pModel, ref modelLen
            );
            if (modelLen <= 0 || modelLen > ModelBytes) modelLen = ModelBytes;
            Marshal.Copy(pModel, model, 0, modelLen);

            Console.Error.WriteLine($"[sdk] fv_extract_model -> ok={ok}, model={modelLen} bytes");
            return (ok, model.Take(modelLen).ToArray());
        }
        finally
        {
            Marshal.FreeHGlobal(p1);
            Marshal.FreeHGlobal(p2);
            Marshal.FreeHGlobal(p3);
            Marshal.FreeHGlobal(pModel);
        }
    }

    // -----------------------------------------------------------------------
    // JSON helpers
    // -----------------------------------------------------------------------
    private static void WriteJson(object o) =>
        Console.WriteLine(JsonSerializer.Serialize(o, new JsonSerializerOptions
        {
            WriteIndented = true,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        }));

    private static void Error(string code, string msg)
    {
        WriteJson(new { ok = false, error = new { code, message = msg } });
        Environment.ExitCode = 1;
    }

    // -----------------------------------------------------------------------
    // Commands
    // -----------------------------------------------------------------------
    private static int CmdEnroll(Sdk sdk, string p1, string p2, string p3, string outPath)
    {
        var r1 = LoadImageRaw(p1);
        var r2 = LoadImageRaw(p2);
        var r3 = LoadImageRaw(p3);
        var (ok, model) = CallExtractModel(sdk, r1, r2, r3);
        if (!ok) { Error("extract_failed", "fv_extract_model returned FALSE"); return 2; }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
        File.WriteAllBytes(outPath, model);
        WriteJson(new
        {
            ok = true,
            operation = "enroll",
            images = new[] { p1, p2, p3 },
            model = new { path = outPath, bytes = model.Length }
        });
        return 0;
    }

    private static int CmdMatch(Sdk sdk, string probePath, string modelPath)
    {
        byte[] raw = LoadImageRaw(probePath);
        byte[] model = File.ReadAllBytes(modelPath);
        if (model.Length != ModelBytes)
        { Error("model_length", $"expected {ModelBytes}, got {model.Length}"); return 1; }

        var (ok, score) = SafeCallMatch(sdk, raw, model);
        WriteJson(new
        {
            ok,
            operation = "match",
            score,
            threshold = sdk.Threshold,
            isMatch = ok && score >= sdk.Threshold
        });
        return ok ? 0 : 2;
    }

    private static int CmdMatch1N(Sdk sdk, string probePath, string modelDir)
    {
        if (!Directory.Exists(modelDir))
        { Error("dir_missing", modelDir); return 1; }

        byte[] raw = LoadImageRaw(probePath);
        byte[] jpeg = EncodeJpeg(raw);
        byte[] enc = EncryptBytes(sdk, jpeg);

        var modelFiles = Directory.EnumerateFiles(modelDir, "*.*", SearchOption.AllDirectories)
            .Where(f =>
            {
                try { return new FileInfo(f).Length == ModelBytes; }
                catch { return false; }
            })
            .OrderBy(f => f)
            .ToList();

        if (modelFiles.Count == 0)
        { Error("no_models", $"no {ModelBytes}-byte model files in {modelDir}"); return 1; }

        // Pre-copy encrypted probe to native buffer (reused for all model files)
        IntPtr pEnc = Marshal.AllocHGlobal(enc.Length);
        IntPtr pModel = IntPtr.Zero;
        Marshal.Copy(enc, 0, pEnc, enc.Length);

        var results = new List<object>();
        float best = float.MinValue;
        string? bestPath = null;
        int errors = 0;

        for (int i = 0; i < modelFiles.Count; i++)
        {
            string fp = modelFiles[i];
            try
            {
                byte[] m = File.ReadAllBytes(fp);
                if (m.Length != ModelBytes) { errors++; continue; }

                // Allocate pModel on first use, reuse for subsequent
                if (pModel == IntPtr.Zero) pModel = Marshal.AllocHGlobal(ModelBytes);
                Marshal.Copy(m, 0, pModel, ModelBytes);

                bool ok = sdk.Match!(pEnc, enc.Length, pModel, ModelBytes, out float score);
                bool hit = score >= sdk.Threshold;
                results.Add(new { path = fp, index = i, ok = true, score, hit });
                if (score > best) { best = score; bestPath = fp; }
            }
            catch (AccessViolationException ave)
            {
                Console.Error.WriteLine($"[sdk] match crashed for {fp}: {ave.Message}");
                results.Add(new { path = fp, index = i, ok = false, score = 0.0f, hit = false, error = "SDK access violation" });
                errors++;
            }
            catch (Exception ex)
            {
                results.Add(new { path = fp, index = i, ok = false, score = 0.0f, hit = false, error = ex.Message });
                errors++;
            }
        }
        Marshal.FreeHGlobal(pEnc);
        if (pModel != IntPtr.Zero) Marshal.FreeHGlobal(pModel);

        // Rank by score descending (sort by the score field, then by index as tiebreaker)
        var ranked = results
            .Select(r => (dynamic)r)
            .OrderByDescending(r => (float)r.score)
            .ThenBy(r => (int)r.index)
            .ToList();

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

    private static int CmdInfo(Sdk sdk)
    {
        WriteJson(new
        {
            ok = true,
            sdk = new
            {
                dll = sdk.DllPath,
                threshold = sdk.Threshold,
                modelBytes = ModelBytes,
                imageSize = $"{ImageW}x{ImageH}",
                imageBytes = ImageBytes
            }
        });
        return 0;
    }

    private static void PrintUsage()
    {
        Console.Error.WriteLine("FVVein — Clean C# finger-vein enrollment + matching");
        Console.Error.WriteLine("(no IplImage, no OpenCV P/Invoke — pure managed .NET)");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Commands:");
        Console.Error.WriteLine("  --enroll  <img1> <img2> <img3> <output.dat>   create enrollment model");
        Console.Error.WriteLine("  --match   <probe> <model_file>                  1:1 match");
        Console.Error.WriteLine("  --match1n <probe> <models_dir>                  1:N identification");
        Console.Error.WriteLine("  --info                                      SDK info");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Image formats: raw 320x240 grayscale (76800 bytes) or JPEG (FF D8)");
    }

    // -----------------------------------------------------------------------
    // Entry point
    // -----------------------------------------------------------------------
    private static int Main(string[] args)
    {
        try
        {
            // Find SDK DLL
            string sdkPath = new[]
            {
                @"D:\WorkSpace\Research\Research\ENG-SDK-C++\DEMO\Release\thin_FingerVein.dll",
                @"D:\WorkSpace\Research\Research\ENG-SDK-C++\DEMO\Release\FingerVein_v3_dev.dll",
            }.FirstOrDefault(File.Exists)
              ?? throw new FileNotFoundException("thin_FingerVein.dll not found in known locations");

            Console.Error.WriteLine($"SDK: {sdkPath}");
            using Sdk sdk = Sdk.Load(sdkPath);

            if (args.Length == 0) { PrintUsage(); return 1; }

            string cmd = args[0];
            int rc;
            if (cmd == "--enroll" && args.Length == 5)
                rc = CmdEnroll(sdk, args[1], args[2], args[3], args[4]);
            else if (cmd == "--match" && args.Length == 3)
                rc = CmdMatch(sdk, args[1], args[2]);
            else if (cmd == "--match1n" && args.Length == 3)
                rc = CmdMatch1N(sdk, args[1], args[2]);
            else if (cmd == "--info")
                rc = CmdInfo(sdk);
            else if (cmd == "-h" || cmd == "--help")
            {
                PrintUsage();
                rc = 0;
            }
            else
            {
                Error("unknown", $"unknown command: {cmd}");
                rc = 1;
            }
            return rc;
        }
        catch (Exception ex)
        {
            Error("exception", $"{ex.GetType().Name}: {ex.Message}");
            return 1;
        }
    }
}
