#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

// Function to convert from MSB first (big-endian) to host format
uint32_t convert_endian(uint32_t value);

#endif // UTILS_H