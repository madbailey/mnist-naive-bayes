#ifndef HOG_H
#define HOG_H

#include <stdint.h>
#include "mnist_loader.h"

// Structure to hold HOG features
typedef struct {
    double *features;     // HOG feature vector
    uint32_t numFeatures; // Number of features per image
    uint32_t numImages;   // Number of images
    uint8_t *labels;      // Labels (copied from original dataset)
} HOGFeatures;

// Extract HOG features from an MNIST dataset
void extractHOGFeatures(MNISTDataset *dataset, HOGFeatures *hogFeatures, 
                        int cellSize, int numBins);

// Free memory allocated for HOG features
void freeHOGFeatures(HOGFeatures *hogFeatures);

#endif // HOG_H