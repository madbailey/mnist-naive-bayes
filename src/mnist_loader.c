
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mnist_loader.h"
#include "utils.h"

// Function to load an MNIST dataset
int loadMNISTDataset(const char *imageFilename, const char *labelFilename, MNISTDataset *dataset) {
    FILE *imageFile, *labelFile;
    uint32_t imageMagic, labelMagic, numLabels;
    
    // Open the image file
    imageFile = fopen(imageFilename, "rb");
    if (imageFile == NULL) {
        perror("Error opening image file");
        return 0;
    }
    
    // Open the label file
    labelFile = fopen(labelFilename, "rb");
    if (labelFile == NULL) {
        perror("Error opening label file");
        fclose(imageFile);
        return 0;
    }
    
    // Read image header
    fread(&imageMagic, sizeof(imageMagic), 1, imageFile);
    fread(&dataset->numImages, sizeof(dataset->numImages), 1, imageFile);
    fread(&dataset->rows, sizeof(dataset->rows), 1, imageFile);
    fread(&dataset->cols, sizeof(dataset->cols), 1, imageFile);
    
    // Read label header
    fread(&labelMagic, sizeof(labelMagic), 1, labelFile);
    fread(&numLabels, sizeof(numLabels), 1, labelFile);
    
    // Convert headers from big-endian
    imageMagic = convert_endian(imageMagic);
    dataset->numImages = convert_endian(dataset->numImages);
    dataset->rows = convert_endian(dataset->rows);
    dataset->cols = convert_endian(dataset->cols);
    
    labelMagic = convert_endian(labelMagic);
    numLabels = convert_endian(numLabels);
    
    // Verify the magic numbers
    if (imageMagic != 2051 || labelMagic != 2049) {
        printf("Invalid file format\n");
        fclose(imageFile);
        fclose(labelFile);
        return 0;
    }
    
    // Check if the number of images matches the number of labels
    if (dataset->numImages != numLabels) {
        printf("Number of images doesn't match number of labels\n");
        fclose(imageFile);
        fclose(labelFile);
        return 0;
    }
    
    // Calculate image size
    dataset->imageSize = dataset->rows * dataset->cols;
    
    // Allocate memory for images and labels
    dataset->images = (uint8_t*)malloc(dataset->numImages * dataset->imageSize);
    dataset->labels = (uint8_t*)malloc(dataset->numImages);
    
    if (dataset->images == NULL || dataset->labels == NULL) {
        printf("Memory allocation failed\n");
        fclose(imageFile);
        fclose(labelFile);
        free(dataset->images);  // Safe to call free on NULL
        free(dataset->labels);
        return 0;
    }
    
    // Read all images
    if (fread(dataset->images, 1, dataset->numImages * dataset->imageSize, imageFile) != 
        dataset->numImages * dataset->imageSize) {
        printf("Failed to read all image data\n");
        fclose(imageFile);
        fclose(labelFile);
        free(dataset->images);
        free(dataset->labels);
        return 0;
    }
    
    // Read all labels
    if (fread(dataset->labels, 1, dataset->numImages, labelFile) != dataset->numImages) {
        printf("Failed to read all label data\n");
        fclose(imageFile);
        fclose(labelFile);
        free(dataset->images);
        free(dataset->labels);
        return 0;
    }
    
    // Close files
    fclose(imageFile);
    fclose(labelFile);
    
    return 1;  // Success
}

// Function to free the dataset
void freeMNISTDataset(MNISTDataset *dataset) {
    free(dataset->images);
    free(dataset->labels);
    dataset->images = NULL;
    dataset->labels = NULL;
}

// Create a new function in mnist_loader.c for loading EMNIST data specifically
int loadEMNISTDataset(const char *imageFilename, const char *labelFilename, MNISTDataset *dataset) {
    // First, load the dataset normally
    if (!loadMNISTDataset(imageFilename, labelFilename, dataset)) {
        return 0; // Return if loading fails
    }
    
    // Then transform each image to correct EMNIST orientation
    printf("Transforming EMNIST images to standard orientation...\n");
    for (uint32_t i = 0; i < dataset->numImages; i++) {
        transformEMNISTImage(&dataset->images[i * dataset->imageSize], dataset->rows, dataset->cols);
        
        // Print progress for large datasets
        if ((i + 1) % 10000 == 0 || i + 1 == dataset->numImages) {
            printf("Transformed %u/%u images\n", i + 1, dataset->numImages);
        }
    }
    
    return 1;
}

void transformEMNISTImageBetter(uint8_t *src, uint32_t size, uint8_t *dst, int rotationAngle) {
    rotationAngle = rotationAngle % 360;
    if (rotationAngle < 0) rotationAngle += 360;

    if (rotationAngle == 0) {
        memcpy(dst, src, size * size);
    }
    else if (rotationAngle == 90) {
        for (uint32_t r = 0; r < size; r++) {
            for (uint32_t c = 0; c < size; c++) {
                dst[c * size + (size - 1 - r)] = src[r * size + c];
            }
        }
    }
    else if (rotationAngle == 180) {
        for (uint32_t r = 0; r < size; r++) {
            for (uint32_t c = 0; c < size; c++) {
                dst[(size - 1 - r) * size + (size - 1 - c)] = src[r * size + c];
            }
        }
    }
    else if (rotationAngle == 270) {
        for (uint32_t r = 0; r < size; r++) {
            for (uint32_t c = 0; c < size; c++) {
                dst[r * size + c] = src[c * size + (size - 1 - r)];
            }
        }
    }
}

// Updated transformEMNISTImage function in mnist_loader.c
void transformEMNISTImage(uint8_t *image, uint32_t rows, uint32_t cols) {
    // Assume square images (e.g., 28x28)
    uint32_t size = rows;  
    uint8_t *temp = (uint8_t*)malloc(size * size);
    if (!temp) {
        printf("Failed to allocate memory for EMNIST transformation\n");
        return;
    }
    
    // Use the improved transformer with the desired rotation angle.
    // Try 270 degrees if that gives the correct orientation.
    transformEMNISTImageBetter(image, size, temp, 270);
    
    memcpy(image, temp, size * size);
    free(temp);
}
// Function to display an image as ASCII art
void displayMNISTImage(uint8_t *image, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            // Convert pixel value to a character based on intensity
            char c = ' ';
            uint8_t pixel = image[i * cols + j];
            if (pixel > 200) c = '#';
            else if (pixel > 150) c = '+';
            else if (pixel > 100) c = '-';
            else if (pixel > 50) c = '.';
            printf("%c", c);
        }
        printf("\n");
    }
}