import ctypes, numpy as np

# Load the DLL
save_lib = ctypes.CDLL(r'./save_direct_io.dll')

# Signatures
save_lib.save_direct_io.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
save_lib.save_direct_io.restype  = ctypes.c_size_t

save_lib.read_raw.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
save_lib.read_raw.restype  = ctypes.c_size_t

# OPTIONAL: extended variants with a direct flag (0=buffered, 1=direct)
save_lib.save_direct_io_ex.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_int]
save_lib.save_direct_io_ex.restype  = ctypes.c_size_t
save_lib.read_raw_ex.argtypes       = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_int]
save_lib.read_raw_ex.restype        = ctypes.c_size_t

# ---- Write (your exact two lines) ----
np_data = np.random.randint(0, 256, size=(10, 608, 1024), dtype=np.uint8)  # example: 10 frames concatenated
filename = b'frames.raw'
data_ptr = np_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
bytes_written = save_lib.save_direct_io(filename, data_ptr, np_data.nbytes)

# ---- Read back ----
# Preallocate destination buffer (same size)
dst = np.empty_like(np_data)
dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
bytes_read = save_lib.read_raw(filename, dst_ptr, dst.nbytes)

# verify
assert bytes_written == np_data.nbytes == bytes_read
assert np.array_equal(np_data, dst)