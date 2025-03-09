#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "mnist_loader.h"
#include <stdint.h>

// Preprocessing options structure
typedef struct {
    int applyNormalization;    // 1 = apply size/position normalization
    int applyThresholding;     // 1 = apply adaptive thresholding
    int applySlantCorrection;  // 1 = apply slant correction
    int applyNoiseRemoval;     // 1 = apply noise removal
    int applyStrokeNorm;       // 1 = apply stroke width normalization
    int applyThinning;         // 1 = apply thinning/skeletonization
    
    // Parameters for different preprocessing steps
    double slantAngleLimit;    // Maximum slant angle to correct (in radians)
    int noiseThreshold;        // Size of noise specks to remove (in pixels)
    int targetStrokeWidth;     // Target stroke width after normalization
    int borderSize;            // Border padding after normalization
} PreprocessingOptions;

// Initialize default preprocessing options
void initDefaultPreprocessing(PreprocessingOptions *options);

// Apply full preprocessing pipeline to a single image
void preprocessImage(uint8_t *image, uint8_t *processedImage, 
                    uint32_t rows, uint32_t cols,
                    PreprocessingOptions *options);

// Process entire dataset with given options
void preprocessDataset(MNISTDataset *dataset, MNISTDataset *processedDataset,
                      PreprocessingOptions *options);

// Individual preprocessing functions that can be called separately:

// Calculate and correct slant angle
double calculateSlant(uint8_t *image, int minX, int minY, int maxX, int maxY,
                     uint32_t rows, uint32_t cols);
void correctSlant(uint8_t *image, uint8_t *result, double slantAngle,
                 uint32_t rows, uint32_t cols);

// Normalize stroke width 
void normalizeStrokeWidth(uint8_t *image, uint8_t *result, int targetWidth,
                         uint32_t rows, uint32_t cols);

// Adaptive thresholding
void adaptiveThreshold(uint8_t *image, uint8_t *result, int windowSize, 
                      double c, uint32_t rows, uint32_t cols);

// Remove small noise specks
void removeNoise(uint8_t *image, uint8_t *result, int threshold,
                uint32_t rows, uint32_t cols);

// Normalization (centering and scaling)
void normalizeSize(uint8_t *image, uint8_t *result, int border,
                  uint32_t rows, uint32_t cols);

// Thinning/skeletonization (Zhang-Suen algorithm)
void thinImage(uint8_t *image, uint8_t *result, uint32_t rows, uint32_t cols);

// Calculate bounding box of character in image
void findBoundingBox(uint8_t *image, uint32_t rows, uint32_t cols,
                    int *minX, int *minY, int *maxX, int *maxY);

#endif // PREPROCESSING_H