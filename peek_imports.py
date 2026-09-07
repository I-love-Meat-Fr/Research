import struct
p = r'D:\WorkSpace\Research\Research\ENG-SDK-C++\DEMO\Release\zhy_fv401.dll'
with open(p, 'rb') as f:
    data = f.read()
pe_off = struct.unpack('<I', data[0x3c:0x40])[0]
opt_off = pe_off + 0x18
magic = struct.unpack('<H', data[opt_off:opt_off+2])[0]
print('Optional header magic:', hex(magic))

num_sections = struct.unpack('<H', data[pe_off+6:pe_off+8])[0]
size_opt_hdr = struct.unpack('<H', data[pe_off+0x14:pe_off+0x16])[0]
sections_off = pe_off + 0x18 + size_opt_hdr

sections = []
for i in range(num_sections):
    s = sections_off + i*40
    name = data[s:s+8].rstrip(b'\x00').decode('latin-1', errors='replace')
    vsize = struct.unpack('<I', data[s+8:s+12])[0]
    va    = struct.unpack('<I', data[s+12:s+16])[0]
    rsize = struct.unpack('<I', data[s+16:s+20])[0]
    roff  = struct.unpack('<I', data[s+20:s+24])[0]
    sections.append((name, va, vsize, roff, rsize))


def rva_to_off(rva):
    for name, va, vsize, roff, rsize in sections:
        if va <= rva < va + max(vsize, rsize):
            return roff + (rva - va)
    return None


dd_off = opt_off + 96
import_rva = struct.unpack('<I', data[dd_off+8:dd_off+12])[0]
print('Import dir RVA:', hex(import_rva))

imp_off = rva_to_off(import_rva)
off = imp_off
dlls = []
while True:
    ilt_rva  = struct.unpack('<I', data[off:off+4])[0]
    name_rva = struct.unpack('<I', data[off+12:off+16])[0]
    if ilt_rva == 0 and name_rva == 0:
        break
    name_off = rva_to_off(name_rva)
    nul = data.index(b'\x00', name_off)
    dll_name = data[name_off:nul].decode('latin-1', errors='replace')
    dlls.append(dll_name)
    off += 20

print('DLLs zhy_fv401.dll depends on:')
for d in dlls:
    print('   -', d)
