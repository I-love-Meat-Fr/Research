// ENG Finger Vein JPG decoder (x86 host for zhy_fv401.dll)
// Usage:
//   DecryptNir <input.jpg> [output.jpg]
// Or to batch-process every <personId>/jpg/m*.jpg under a root:
//   DecryptNir <model_root> <output_root>

// History:
//   v2 - added SetDllDirectoryW before LoadLibraryW so transitive deps
//        (msvcr100.dll, msvcp100.dll, iphlpapi.dll) resolve from the same
//        folder as zhy_fv401.dll. Required on x64 Windows where SysWOW64
//        does not contain VC++ 2010 redistributable but the SDK folder does.

using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

internal static class Program
{
    // Fixed 32-byte header prepended to every captured .jpg by the SDK.
    private static readonly byte[] FixedHeader = new byte[]
    {
        0x2D, 0x6A, 0x00, 0xDD, 0xD7, 0xC5, 0xAE, 0xD7,
        0x47, 0x0E, 0xFF, 0x72, 0x62, 0x52, 0x91, 0xB8,
        0xEE, 0x07, 0x94, 0x78, 0xFB, 0x1E, 0x75, 0x5E,
        0x0E, 0x4A, 0x8D, 0x2D, 0x87, 0xB8, 0xB5, 0x82,
    };

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr LoadLibraryW(string lpFileName);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetDllDirectoryW(string? lpPathName);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr hModule, string lpProcName);

    private delegate int ZhyDecryptDelegate(IntPtr dst, IntPtr src, int length);

    private static ZhyDecryptDelegate? LoadDecrypt(string dllPath)
    {
        // Resolve transitive deps (msvcr100.dll, msvcp100.dll, iphlpapi.dll)
        // from the same folder as the DLL we're about to load. Reset to NULL
        // after the call to restore the default search order.
        string? dir = Path.GetDirectoryName(dllPath);
        if (!string.IsNullOrEmpty(dir))
        {
            SetDllDirectoryW(dir);
        }

        IntPtr h;
        try
        {
            h = LoadLibraryW(dllPath);
        }
        finally
        {
            SetDllDirectoryW(null);
        }

        if (h == IntPtr.Zero)
        {
            int err = Marshal.GetLastWin32Error();
            throw new InvalidOperationException($"LoadLibraryW({dllPath}) failed with 0x{err:X}");
        }

        IntPtr proc = GetProcAddress(h, "ZHY_Decrypt");
        if (proc == IntPtr.Zero)
        {
            Console.Error.WriteLine($"  ZHY_Decrypt not in {dllPath}, retrying with fv_Decrypt");
            proc = GetProcAddress(h, "fv_Decrypt");
        }
        if (proc == IntPtr.Zero)
        {
            throw new InvalidOperationException("Neither ZHY_Decrypt nor fv_Decrypt found in " + dllPath);
        }
        return (ZhyDecryptDelegate)Marshal.GetDelegateForFunctionPointer(proc, typeof(ZhyDecryptDelegate));
    }

    private static byte[] Decrypt(ZhyDecryptDelegate decrypt, byte[] src)
    {
        int len = src.Length;
        int aligned = (len + 15) / 16 * 16;

        IntPtr srcBuf = Marshal.AllocHGlobal(aligned);
        IntPtr dstBuf = Marshal.AllocHGlobal(aligned);
        try
        {
            Marshal.Copy(src, 0, srcBuf, len);
            for (int i = len; i < aligned; i++) Marshal.WriteByte(srcBuf, i, 0);

            int rc = decrypt(dstBuf, srcBuf, len);
            if (rc != 0)
                throw new InvalidOperationException($"ZHY_Decrypt returned {rc}");

            byte[] dst = new byte[len];
            Marshal.Copy(dstBuf, dst, 0, len);
            return dst;
        }
        finally
        {
            Marshal.FreeHGlobal(srcBuf);
            Marshal.FreeHGlobal(dstBuf);
        }
    }

    private static void DecodeOne(ZhyDecryptDelegate decrypt, string inPath, string outPath)
    {
        byte[] raw = File.ReadAllBytes(inPath);
        bool hasHeader = raw.Length >= 32 && raw.Take(32).SequenceEqual(FixedHeader);
        byte[] payload = hasHeader ? raw : raw;  // try decrypting full file
        Console.WriteLine($"  [hdr] header present: {hasHeader}, payload {payload.Length} B");

        byte[] plaintext = Decrypt(decrypt, payload);
        Console.WriteLine($"  [dec] plaintext {plaintext.Length} B, magic {plaintext[0]:X2} {plaintext[1]:X2}, tail ...{plaintext[plaintext.Length-2]:X2} {plaintext[plaintext.Length-1]:X2}");

        Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
        File.WriteAllBytes(outPath, plaintext);
        Console.WriteLine($"  [out] {outPath}");
    }

    private static void DecodeFolder(ZhyDecryptDelegate decrypt, string rootIn, string rootOut)
    {
        foreach (string jpgPath in Directory.EnumerateFiles(rootIn, "m*.jpg", SearchOption.AllDirectories))
        {
            string rel = Path.GetRelativePath(rootIn, jpgPath);
            string outPath = Path.Combine(rootOut, rel);
            Console.WriteLine($"\n{jpgPath}");
            try
            {
                DecodeOne(decrypt, jpgPath, outPath);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"  ERROR: {ex.Message}");
            }
        }
    }

    private static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine("Usage:");
            Console.Error.WriteLine("  DecryptNir <input.jpg> [output.jpg]");
            Console.Error.WriteLine("  DecryptNir <model_root> <output_root>");
            return 1;
        }

        string[] candidates = {
            @"D:\WorkSpace\Research\Research\ENG-SDK-C++\DEMO\Release\zhy_fv401.dll",
            @"D:\WorkSpace\Research\Research\ENG-SDK-C++\SDK Dynamic library\zhy_fv401.dll",
            @"D:\WorkSpace\Research\Research\ENG-SDK-C++\DEMO\Release\thin_FingerVein.dll",
            @"D:\WorkSpace\Research\Research\ENG-SDK-C++\DEMO\Release\FingerVein_v3_dev.dll",
        };

        ZhyDecryptDelegate? decrypt = null;
        foreach (string p in candidates)
        {
            if (!File.Exists(p)) continue;
            Console.WriteLine($"Trying {p}");
            try
            {
                decrypt = LoadDecrypt(p);
                Console.WriteLine($"  OK");
                break;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"  failed: {ex.Message}");
            }
        }
        if (decrypt == null)
        {
            Console.Error.WriteLine("No DLL could be loaded.");
            return 2;
        }

        if (Directory.Exists(args[0]))
        {
            string outRoot = args.Length > 1 ? args[1] : Path.Combine(Path.GetDirectoryName(args[0])!, "decoded");
            DecodeFolder(decrypt, args[0], outRoot);
        }
        else
        {
            string outPath = args.Length > 1 ? args[1] : Path.Combine(Path.GetDirectoryName(args[0])!, Path.GetFileNameWithoutExtension(args[0]) + "_dec.jpg");
            DecodeOne(decrypt, args[0], outPath);
        }

        Console.WriteLine("\nDone.");
        return 0;
    }
}
