#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "hog.h"

static void computeGradient(uint8_t *image, uint32_t rows, uint32_t cols, 
    int x, int y, double *magnitude, double *orientation) {
    
    double dx, dy;

    // Handle image boundaries with mirroring
    uint32_t left = (x > 0) ? (uint32_t)(x - 1) : 0;
    uint32_t right = (x < (int)(cols - 1)) ? (uint32_t)(x + 1) : cols - 1;
    uint32_t top = (y > 0) ? (uint32_t)(y - 1) : 0;
    uint32_t bottom = (y < (int)(rows - 1)) ? (uint32_t)(y + 1) : rows - 1;

    // Compute gradients using central differences
    dx = (double)image[y * cols + right] - (double)image[y * cols + left];
    dy = (double)image[bottom * cols + x] - (double)image[top * cols + x];

    // Calculate magnitude
    *magnitude = sqrt(dx * dx + dy * dy);
    
    // Handle the case when gradient is zero (avoid undefined orientation)
    if (fabs(dx) < 1e-6 && fabs(dy) < 1e-6) {
        *orientation = 0;
    } else {
        // Calculate orientation in the range [-pi, pi]
        *orientation = atan2(dy, dx);
        
        // Convert orientation to degrees in the range [0, 180)
        *orientation = fmod((*orientation * 180.0 / M_PI) + 180.0, 180.0);
    }
}

void extractHOGFeatures(MNISTDataset *dataset, HOGFeatures *hogFeatures, int cellSize, int numBins) {
    //calculate number of cells in each direction
    int cellsX = dataset->cols / cellSize;
    int cellsY = dataset->rows / cellSize;

    //calculate the number of hog features in the image
    hogFeatures->features = (double*)malloc(hogFeatures->numImages * hogFeatures->numFeatures * sizeof(double));

    // Only allocate and copy labels if we have them in the dataset
    if (dataset->labels != NULL) {
        hogFeatures->labels = (uint8_t*)malloc(hogFeatures->numImages * sizeof(uint8_t));
        if (hogFeatures->labels == NULL) {
            printf("failed to allocate memory for HOG labels\n");
            free(hogFeatures->features);
            hogFeatures->features = NULL;
            return;
        }
        memcpy(hogFeatures->labels, dataset->labels, hogFeatures->numImages * sizeof(uint8_t));
    } else {
        hogFeatures->labels = NULL;  // Explicitly set to NULL if no labels
    }

    if (hogFeatures->features == NULL) {
        printf("failed to allocate memory for HOG features\n");
        return;
    }

    // Initialize all features to zero
    memset(hogFeatures->features, 0, hogFeatures->numImages * hogFeatures->numFeatures * sizeof(double));
    
    // Process images
    for (uint32_t imgIdx = 0; imgIdx < dataset->numImages; imgIdx++) {
        uint8_t *image = &dataset->images[imgIdx * dataset->imageSize];
        double *imgFeatures = &hogFeatures->features[imgIdx * hogFeatures->numFeatures];
        
        // process cell
        for (int cy = 0; cy < cellsY; cy++) {
            for (int cx = 0; cx < cellsX; cx++) {
                // make histogram for this cell (one bin for each orientation range)
                double histogram[numBins];
                memset(histogram, 0, numBins * sizeof(double));
                
                // Process each pixel in the cell
                for (int y = cy * cellSize; y < (cy + 1) * cellSize; y++) {
                    for (int x = cx * cellSize; x < (cx + 1) * cellSize; x++) {
                        double magnitude, orientation;
                        computeGradient(image, dataset->rows, dataset->cols, x, y, 
                                    &magnitude, &orientation);
                        
                        // Determine which bin the orientation falls into
                        int bin = (int)(orientation * numBins / 180.0);
                        if (bin >= numBins) bin = numBins - 1; // Safety check
                        
                        // Add weighted magnitude to the histogram
                        histogram[bin] += magnitude;
                    }
                }
                
                // Normalize the histogram and store in feature vector
                double sum = 0.0;
                for (int b = 0; b < numBins; b++) {
                    sum += histogram[b] * histogram[b];
                }
                double norm = sqrt(sum + 1e-6); // Avoid division by zero
                
                // Store normalized histogram in feature vector
                int featureOffset = (cy * cellsX + cx) * numBins;
                for (int b = 0; b < numBins; b++) {
                    imgFeatures[featureOffset + b] = histogram[b] / norm;
                }
            }
        }
        
        // Print progress
        if ((imgIdx + 1) % 10000 == 0 || imgIdx + 1 == dataset->numImages) {
            printf("Processed %u/%u images\n", imgIdx + 1, dataset->numImages);
        }
    }

    printf("Extracted HOG features: %u images, %u features per image\n", 
        hogFeatures->numImages, hogFeatures->numFeatures);
}

void freeHOGFeatures(HOGFeatures *hogFeatures) {
    if (hogFeatures) {
        if (hogFeatures->features) {
            free(hogFeatures->features);
            hogFeatures->features = NULL;
        }
        
        if (hogFeatures->labels) {
            free(hogFeatures->labels);
            hogFeatures->labels = NULL;
        }
    }
}