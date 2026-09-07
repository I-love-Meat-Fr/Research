# Deep analysis of the data format
import os
import numpy as np

# Load dat file
dat_path = r'D:\WorkSpace\Research\Research\model\1094\1.dat'
with open(dat_path, 'rb') as f:
    dat = np.frombuffer(f.read(), dtype=np.uint8)

print(f'DAT file size: {len(dat)} bytes')
print(f'Statistics: min={dat.min()}, max={dat.max()}, mean={dat.mean():.1f}, std={dat.std():.1f}')

# Convert to bytes for hex display
dat_bytes = dat.tobytes()
print(f'\nFirst 64 bytes: {dat_bytes[:64].hex()}')
print(f'Last 64 bytes: {dat_bytes[-64:].hex()}')

# Check histogram
hist, bins = np.histogram(dat, bins=16)
print(f'\nHistogram (16 bins):')
for i in range(len(hist)):
    low = int(bins[i])
    high = int(bins[i+1])
    bar = '#' * (hist[i] // 50)
    print(f'  {low:3d}-{high:3d}: {bar} ({hist[i]})')

# Check if it's 8-bit grayscale image data
# 3352 = 16 * 209 = 8 * 419 = 4 * 838 = 2 * 1676
# Common camera resolutions: 640x480=307200, 320x240=76800, etc.
# 3352 is not a standard image size

# Try different aspect ratios
print('\nPossible image dimensions:')
candidates = []
for h in range(1, 100):
    w = len(dat) // h
    if w * h == len(dat) and 1.0 < w/h < 4.0:
        candidates.append((h, w, w/h))

for h, w, ratio in candidates[:10]:
    print(f'  {h}x{w} (ratio {ratio:.2f})')

# Check if the data might be compressed with a different algorithm
# Try LZ4
try:
    import lz4.block
    dec = lz4.block.decompress(dat.tobytes())
    print(f'\nLZ4 decompress: {len(dec)} bytes')
except:
    print('\nLZ4 not available or failed')

# Try zstandard
try:
    import zstandard as zstd
    dec = zstd.decompress(dat.tobytes())
    print(f'Zstd decompress: {len(dec)} bytes')
except:
    pass

# Try LZMA
try:
    import lzma
    dec = lzma.decompress(dat.tobytes())
    print(f'LZMA decompress: {len(dec)} bytes')
except:
    pass

# Try bz2
try:
    import bz2
    dec = bz2.decompress(dat.tobytes())
    print(f'BZ2 decompress: {len(dec)} bytes')
except:
    pass
