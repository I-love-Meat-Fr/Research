# FVVein — C# Finger Vein Enrollment & Matching Tool

Clean rewrite of the old `FVMatcher` — no `IplImage` struct, no OpenCV P/Invoke.

## What was removed

- ❌ Custom `IplImageHeader` struct (x86 layout was broken)
- ❌ OpenCV 2.4.3 P/Invoke (`cvLoadImage`, `cvEncodeImage`, `cvCreateImage`, `cvSetData`, `cvReleaseImage`)
- ❌ All `[MarshalAs(UnmanagedType.LPArray, SizeParamIndex = ...)]` annotations on byte arrays (caused `MarshalDirectiveException`)

## What replaced it

- ✅ `System.Drawing.Bitmap` for JPEG encode/decode (pure managed, no interop)
- ✅ `Marshal.AllocHGlobal` + `Marshal.Copy` for SDK byte buffer interop (correct, deterministic)
- ✅ Structured JSON output, safe error handling

## Status

**The Python E2E pipeline (`tools/FVVeinPipeline/`) is the recommended path.** It is fully functional, doesn't depend on the proprietary SDK, and uses ResNet18 + ArcFace deep learning.

The C# tool is kept as a thin wrapper around `thin_FingerVein.dll` for use cases where the SDK's model files (e.g. `model/1094/1.dat`) are required.

## Known issue: SDK model/probe format coupling

The `thin_FingerVein.dll` SDK crashes (`STATUS_ACCESS_VIOLATION`) when the probe JPEG produced by `System.Drawing` does not exactly match the JPEG layout used during `fv_extract_model`. The proprietary DLL validates the JPEG header structure in a way we cannot reproduce without OpenCV's `cvEncodeImage`.

**Workarounds:**
1. **Recommended**: Use the Python pipeline — no SDK dependency, full pipeline works end-to-end
2. If you must use the C# tool, only pair probe/model combinations that came from the same encoder (e.g. all from OpenCV's `cvEncodeImage`)

## Build

```bash
cd tools/FVVein
dotnet publish --configuration Release -r win-x86 --self-contained -p:PublishSingleFile=true -o publish
```

Output: `publish/FVVein.exe` (single file, self-contained, x86).

## Commands

```
FVVein --enroll  <img1> <img2> <img3> <output.dat>
FVVein --match   <probe> <model_file>
FVVein --match1n <probe> <models_dir>
FVVein --info
```

## File structure

```
tools/FVVein/
├── Program.cs           ← single-file implementation
├── FVVein.csproj        ← net8.0-windows, x86, System.Drawing.Common
└── publish/
    └── FVVein.exe       ← single-file published binary
```
