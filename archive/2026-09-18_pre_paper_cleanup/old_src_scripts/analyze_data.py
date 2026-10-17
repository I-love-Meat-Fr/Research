# Analyze the fixed header
import os
import struct

# Check if the jpg files can be decoded
jpg_path = r'D:\WorkSpace\Research\Research\model\98\jpg\2.jpg'
with open(jpg_path, 'rb') as f:
    jpg_data = f.read()

print(f'jpg file size: {len(jpg_data)} bytes')
print(f'First 32 bytes (header): {jpg_data[:32].hex()}')

# Check payload
payload = jpg_data[32:]
print(f'Payload first 100 bytes: {payload[:100].hex()}')

# Try different decryption approaches
print('\nTrying to find actual image data...')

# Check if there's a known image format embedded
# by looking for common markers
known_markers = {
    0xFFD8FF: 'JPEG Start',
    0xFFD9: 'JPEG End',
    0x89504E47: 'PNG',
    0x49492A00: 'TIFF LE',
    0x4D4D002A: 'TIFF BE',
    0x424D: 'BMP',
}

# Check first 4 bytes of payload
first4 = (payload[0] << 24) | (payload[1] << 16) | (payload[2] << 8) | payload[3]
print(f'\nPayload first 4 bytes as int: 0x{first4:08X}')

# Try XOR with different keys on first 32 bytes of payload
print('\nTrying XOR keys on payload:')
for key in [0x42, 0xAA, 0x55, 0xFF, 0x00]:
    decoded = bytes([b ^ key for b in payload[:32]])
    if decoded[:2] == b'\xff\xd8':
        print(f'  Key 0x{key:02X}: Looks like JPEG!')
        print(f'    {decoded[:16].hex()}')
    elif decoded[:4] == b'\x89PNG':
        print(f'  Key 0x{key:02X}: Looks like PNG!')
        print(f'    {decoded[:16].hex()}')
    else:
        print(f'  Key 0x{key:02X}: {decoded[:16].hex()}')
