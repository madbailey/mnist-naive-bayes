#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

//unsigned 32-bit endian swapper 
uint32_t convert_endian(uint32_t value) {
    return ((value >> 24) & 0xff) | // Move byte 3 to byte 0
           ((value << 8) & 0xff0000) | // Move byte 1 to byte 2
           ((value >> 8) & 0xff00) | // Move byte 2 to byte 1
           ((value << 24) & 0xff000000); // Move byte 0 to byte 3
}

int main() {
    FILE * imageFile;
    uint32_t magic, numImages, numRows, numCols;
    uint8_t *imageData;

    //open the image file
    imageFile = fopen("data/t10k-images.idx3-ubyte", "rb");
    if (imageFile == NULL) {
        perror("Error opening MNIST image file");
        return 1;
    }

    //look at the header
    fread(&magic, sizeof(magic), 1, imageFile);
    fread(&numImages, sizeof(numImages), 1, imageFile);
    fread(&numRows, sizeof(numRows), 1, imageFile);
    fread(&numCols, sizeof(numCols), 1, imageFile);

    // convert MSB to something legible
    magic = convert_endian(magic);
    numImages = convert_endian(numImages);
    numRows = convert_endian(numRows);
    numCols = convert_endian(numCols);

    printf("Magic: %u\n", magic);
    printf("Number of images: %u\n", numImages);
    printf("Number of rows: %u\n", numRows);
    printf("Number of columns: %u\n", numCols);

    //now look at the image data
    imageData = (uint8_t*)malloc(numRows * numCols);
    fread(imageData, 1, numRows * numCols, imageFile);

    fclose(imageFile);

    // Visualize the first image as ASCII art
    printf("First image:\n");
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            // Convert pixel value to a character based on intensity
            char c = ' ';
            uint8_t pixel = imageData[i * numCols + j];
            if (pixel > 200) c = '#';
            else if (pixel > 150) c = '+';
            else if (pixel > 100) c = '-';
            else if (pixel > 50) c = '.';
            printf("%c", c);
        }
        printf("\n");
    }

    free(imageData);
    return 0;
}
