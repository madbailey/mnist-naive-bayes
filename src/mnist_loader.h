#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdint.h>

// Structure to hold our dataset
typedef struct {
    uint8_t *images;       // All images in one continuous array
    uint8_t *labels;       // All labels in one array
    uint32_t numImages;    // Number of images/labels
    uint32_t imageSize;    // Size of each image (rows*cols)
    uint32_t rows;         // Number of rows in each image
    uint32_t cols;         // Number of columns in each image
} MNISTDataset;

// Function to load an MNIST dataset
int loadMNISTDataset(const char *imageFilename, const char *labelFilename, 
                    MNISTDataset *dataset);

// Function to free the dataset
void freeMNISTDataset(MNISTDataset *dataset);

// Function to display an image as ASCII art
void displayMNISTImage(uint8_t *image, uint32_t rows, uint32_t cols);

#endif // MNIST_LOADER_H