// Function to convert from MSB first (big-endian) to host format
uint32_t convert_endian(uint32_t value) {
    return ((value >> 24) & 0xff) | 
           ((value << 8) & 0xff0000) | 
           ((value >> 8) & 0xff00) | 
           ((value << 24) & 0xff000000);
}