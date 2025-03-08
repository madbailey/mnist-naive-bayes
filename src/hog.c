#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "hog.h"

static void computeGradient(__uint8_t *image, __uint32_t rows, __uint32_t cols, 
    int x, int y, double *magnitude, double *orientation) {
    
    double dx, dy;


    int left = (x >0) ? x-1 : 0;
    int right = (x < cols -1) ? x+ 1 : cols -1;
    int top = (y > 0) ? y -1 : 0;
    int bottom = (y < rows -1) ? y + 1 : rows -1;

    //compute gradients using central differences
    dx = (double)image[y * cols + right] - (double)image[y * cols + left];
    dy = (double)image[bottom * cols + x] - (double)image[top * cols + x];

    //calculate magintude and orientation
    *magnitude = sqrt(dx * dx + dy *dy);
    *orientation = atan2(dy, dx);

    //convert orientation to degrees
    *orientation = fmod((*orientation * 180.0 / M_PI) + 180.0, 180.0);
}

void extractHOGFeatures(MNISTDataset *dataset, HOGFeatures *hogFeatures, int cellSize, int numBins) {
    //calculat number of cells in each direction
    int cellsX = dataset->cols /cellSize;
    int cellsY  = dataset->rows /cellSize;

    //calculate the number of hohg features in the image
    hogFeatures->features = (double*)malloc(hogFeatures->numImages * hogFeatures->numFeatures * sizeof(double));

    hogFeatures->labels = (uint8_t*)malloc(hogFeatures->numImages *sizeof(uint8_t));

    if (hogFeatures->features == NULL || hogFeatures->labels == NULL) {
        printf("failed to allocate memory for HOG features\n");
        free(hogFeatures->features);
        free(hogFeatures->labels);
        return;
    }

    memcpy(hogFeatures->labels, dataset->labels, hogFeatures->numImages * sizeof(uint8_t));

    memset(hogFeatures->features, 0, hogFeatures->numImages * sizeof(uint8_t));
    
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