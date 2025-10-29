#include <iostream>
#include <fstream>
#include <cstdint>  // For uint8_t
#include <cstdio>   // For C-style file I/O (fopen, fwrite, fread)

// Use extern "C" to ensure C linkage (no name mangling)
// This is critical for ctypes to find the functions.
extern "C" {

    /**
     * @brief Writes a raw data buffer to a file.
     * * This function matches the signature for:
     * save_lib.save_direct_io(filename.encode('utf-8'), data_ptr, np_data.nbytes)
     * * @param filename   A UTF-8 encoded string for the file path.
     * @param data       A const pointer to the start of the data buffer.
     * @param num_bytes  The total number of bytes to write.
     * @return           The number of bytes successfully written, or -1 on error.
     */
    __declspec(dllexport) long long save_direct_io(const char* filename, const void* data, size_t num_bytes) {
        // Open the file in binary write mode ("wb")
        // Using C-style FILE* is often simpler and faster for raw binary I/O.
        FILE* f = fopen(filename, "wb");
        if (f == nullptr) {
            perror("save_direct_io: Error opening file for writing");
            return -1; // Return -1 to indicate an error
        }

        // Write the data
        // fwrite(ptr, size_of_one_element, num_elements, file_handle)
        size_t bytes_written = fwrite(data, 1, num_bytes, f);

        // Close the file
        fclose(f);

        if (bytes_written != num_bytes) {
            // This indicates a partial write or error
            fprintf(stderr, "save_direct_io: Warning: partial write\n");
            return static_cast<long long>(bytes_written);
        }

        return static_cast<long long>(bytes_written);
    }

    /**
     * @brief Reads raw data from a file directly into a pre-allocated buffer.
     * * @param filename     A UTF-8 encoded string for the file path.
     * @param buffer       A non-const pointer to the buffer to be filled.
     * @param buffer_size  The size of the buffer (and max bytes to read).
     * @return             The number of bytes successfully read, or -1 on error.
     */
    __declspec(dllexport) long long read_raw_data(const char* filename, void* buffer, size_t buffer_size) {
        // Open the file in binary read mode ("rb")
        FILE* f = fopen(filename, "rb");
        if (f == nullptr) {
            perror("read_raw_data: Error opening file for reading");
            return -1; // Return -1 to indicate an error
        }

        // Read data directly into the provided buffer
        size_t bytes_read = fread(buffer, 1, buffer_size, f);

        // Close the file
        fclose(f);

        if (bytes_read == 0 && !feof(f)) {
            // This indicates a read error
            fprintf(stderr, "read_raw_data: Error during file read\n");
            return -1;
        }

        return static_cast<long long>(bytes_read);
    }
}
