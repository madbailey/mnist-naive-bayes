#include "normalization.h"
#include "mnist_loader.h"
#include <stdio.h>  // For debugging (printf)
#include <stdlib.h> // For malloc, calloc, and free
#include <math.h>   // For fabs, sin, cos, tan, round
#include <string.h> // for memcpy

void initDefaultPreprocessing(PreprocessingOptions *options) {
    if (options == NULL) {
        return;  // Handle null pointer
    }
    options->applyNormalization = 1;
    options->applyThresholding = 1;
    options->applySlantCorrection = 1;
    options->applyNoiseRemoval = 1;
    options->applyStrokeNorm = 0; // Stroke normalization off by default
    options->applyThinning = 0;     // Thinning off by default

    options->slantAngleLimit = 0.5;  // radians
    options->noiseThreshold = 2;      // pixels
    options->targetStrokeWidth = 3;
    options->borderSize = 2;          // pixels
}



void findBoundingBox(uint8_t *image, uint32_t rows, uint32_t cols,
                    int *minX, int *minY, int *maxX, int *maxY) {

    if (image == NULL || minX == NULL || minY == NULL || maxX == NULL || maxY == NULL) {
        return; // Handle null pointers
    }

    *minX = cols;
    *minY = rows;
    *maxX = 0;
    *maxY = 0;

    for (uint32_t y = 0; y < rows; ++y) {
        for (uint32_t x = 0; x < cols; ++x) {
            if (image[y * cols + x] > 0) { // Assuming non-zero pixel is part of character
                if ((int)x < *minX) *minX = x;
                if ((int)y < *minY) *minY = y;
                if ((int)x > *maxX) *maxX = x;
                if ((int)y > *maxY) *maxY = y;
            }
        }
    }
}

double calculateSlant(uint8_t *image, int minX, int minY, int maxX, int maxY,
                     uint32_t rows, uint32_t cols) {

    if (image == NULL) return 0.0;  // Handle null pointer

    double shearMoment = 0.0;
    double verticalMoment = 0.0;
     // Calculate moments within bounding box

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            if (image[y * cols + x] > 0) { // Consider only foreground pixels
                shearMoment += (x - (minX + maxX) / 2.0) * (y - (minY + maxY) / 2.0);
                verticalMoment += pow(y - (minY + maxY) / 2.0, 2);
            }
        }
    }

    if(verticalMoment == 0) return 0.0; //Prevent divide-by-zero.
    return -shearMoment / verticalMoment;
}



void correctSlant(uint8_t *image, uint8_t *result, double slantAngle,
                 uint32_t rows, uint32_t cols) {
    if (image == NULL || result == NULL) {
        return; // Handle null pointers
    }

    // Initialize result to 0 (background)
    memset(result, 0, rows * cols);

    double shear = -tan(slantAngle); // Negative for correction

    // Iterate over the *original* image to avoid shearing artifacts
    for (uint32_t y = 0; y < rows; ++y) {
        for (uint32_t x = 0; x < cols; ++x) {
            //calculate the source position
            double srcX = x;
            double srcY = y;

            // Calculate sheared coordinates
            int newX = (int)round(srcX + shear * (srcY - (rows / 2.0)));
            int newY = (int)srcY;  // Y remains unchanged

            // Check bounds for new coordinates
            if (newX >= 0 && newX < (int)cols && newY >= 0 && newY < (int)rows) {
                result[newY * cols + newX] = image[(int)y * cols + (int)x];
            }
        }
    }
}



// Normalize stroke width
void normalizeStrokeWidth(uint8_t *image, uint8_t *result, int targetWidth,
                         uint32_t rows, uint32_t cols) {
    // For simplicity, we'll implement a basic approach using dilation/erosion
    // A more sophisticated approach might estimate actual stroke width first
    
    // Temporary buffer for intermediate results
    uint8_t *temp = (uint8_t*)malloc(rows * cols);
    if (!temp) {
        memcpy(result, image, rows * cols);  // Fallback if allocation fails
        return;
    }
    
    // Make a binary copy of the image first (0 or 255)
    for (uint32_t i = 0; i < rows * cols; i++) {
        temp[i] = (image[i] > 128) ? 255 : 0;
    }
    
    // If targetWidth > 1, perform dilation
    if (targetWidth > 1) {
        // Simple dilation implementation
        memcpy(result, temp, rows * cols);
        
        for (int iter = 0; iter < targetWidth-1; iter++) {
            // Copy result to temp for this iteration
            memcpy(temp, result, rows * cols);
            
            for (uint32_t y = 0; y < rows; y++) {
                for (uint32_t x = 0; x < cols; x++) {
                    // If any neighbor is set, set this pixel
                    if (temp[y * cols + x] == 0) {
                        // Check 4-connected neighbors
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                if (dx == 0 && dy == 0) continue;  // Skip self
                                
                                int ny = y + dy;
                                int nx = x + dx;
                                
                                if (ny >= 0 && ny < (int)rows && nx >= 0 && nx < (int)cols) {
                                    if (temp[ny * cols + nx] > 0) {
                                        result[y * cols + x] = 255;
                                        goto next_pixel;  // Break out of both loops
                                    }
                                }
                            }
                        }
                        next_pixel:;
                    }
                }
            }
        }
    } else {
        // If targetWidth <= 1, use the binary image directly
        memcpy(result, temp, rows * cols);
    }
    
    free(temp);
}



void adaptiveThreshold(uint8_t *image, uint8_t *result, int windowSize,
                      double c, uint32_t rows, uint32_t cols) {

    if (image == NULL || result == NULL || windowSize <= 0) {
        return; // Handle invalid input
    }
    if (windowSize % 2 == 0) windowSize++; // Ensure windowSize is odd.

    for (uint32_t y = 0; y < rows; ++y) {
        for (uint32_t x = 0; x < cols; ++x) {
            // Calculate window boundaries (handle edges)
            int xStart = (int)x - windowSize / 2;
            int yStart = (int)y - windowSize / 2;
            int xEnd = (int)x + windowSize / 2;
            int yEnd = (int)y + windowSize / 2;


            xStart = (xStart < 0) ? 0 : xStart;
            yStart = (yStart < 0) ? 0 : yStart;
            xEnd = (xEnd >= (int)cols) ? cols -1 : xEnd;
            yEnd = (yEnd >= (int)rows) ? rows - 1: yEnd;

            // Calculate mean intensity within the window
            double sum = 0;
            int count = 0;
            for (int wy = yStart; wy <= yEnd; ++wy) {
                for (int wx = xStart; wx <= xEnd; ++wx) {
                    sum += image[wy * cols + wx];
                    count++;
                }
            }
            double mean = (count > 0) ? sum / count : 0;


            // Apply threshold
            result[y * cols + x] = (image[y * cols + x] > (mean - c)) ? 255 : 0;
        }
    }
}



void removeNoise(uint8_t *image, uint8_t *result, int threshold,
                uint32_t rows, uint32_t cols) {

    if (image == NULL || result == NULL) {
        return; // Handle null pointers
    }

    memcpy(result, image, rows * cols); // Start with a copy

    // Iterate through all pixels
    for (uint32_t y = 0; y < rows; y++) {
        for (uint32_t x = 0; x < cols; x++) {
            //if the pixel is 0 we don't need to check it
            if(result[y * cols + x] == 0) continue;

            // If the current pixel is 'on' (e.g., 255), check its neighbors
            int count = 0;  // Count of neighboring 'on' pixels
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue; // Skip the current pixel itself

                    int nx = (int)x + dx;
                    int ny = (int)y + dy;

                    // Check boundary conditions
                    if (nx >= 0 && nx < (int)cols && ny >= 0 && ny < (int)rows) {
                        if (result[ny * cols + nx] > 0) {
                            count++;
                        }
                    }
                }
            }
            // If it has few neighbors (isolated), remove it (set to 0)
            if (count <= threshold) {
                result[y * cols + x] = 0;
            }
        }
    }
}

void normalizeSize(uint8_t *image, uint8_t *result, int border,
                  uint32_t rows, uint32_t cols) {

    if (image == NULL || result == NULL) {
        return; // Handle null pointers
    }

    int minX, minY, maxX, maxY;
    findBoundingBox(image, rows, cols, &minX, &minY, &maxX, &maxY);

    int originalWidth = maxX - minX + 1;
    int originalHeight = maxY - minY + 1;

    // Initialize result to 0 (background)
    memset(result, 0, rows * cols);
    
    // Handle cases where the image is entirely empty (no foreground pixels)
    if (originalWidth <= 0 || originalHeight <= 0) {
        return; // Nothing to normalize
    }

    // Calculate scaling factor (preserve aspect ratio)
    double scaleX = (double)(cols - 2 * border) / originalWidth;
    double scaleY = (double)(rows - 2 * border) / originalHeight;
    double scale = (scaleX < scaleY) ? scaleX : scaleY;

    // Calculate the new dimensions.
    int newWidth = (int)(originalWidth * scale);
    int newHeight = (int)(originalHeight * scale);


    // Calculate centering offset
    int offsetX = (cols - newWidth) / 2;
    int offsetY = (rows - newHeight) / 2;

    // Copy and scale the image
    for (int y = 0; y < originalHeight; y++) {
        for (int x = 0; x < originalWidth; x++) {
            if (image[(minY + y) * cols + (minX + x)] > 0) { // Only process foreground pixels
                int newX = (int)(x * scale) + offsetX;
                int newY = (int)(y * scale) + offsetY;

                if (newX >= 0 && newX < (int)cols && newY >= 0 && newY < (int)rows) {
                    result[newY * cols + newX] = 255; // Set to foreground (255)
                }
            }
        }
    }
}


void thinImage(uint8_t *image, uint8_t *result, uint32_t rows, uint32_t cols) {

    // This implements the Zhang-Suen thinning algorithm.

    if (!image || !result) return;

    memcpy(result, image, rows * cols);

    int changed;
    do {
        changed = 0;
        uint8_t *temp = (uint8_t *)malloc(rows * cols);
        if (!temp) return; // Allocation failed
        memcpy(temp, result, rows * cols);

        // --- First sub-iteration ---
        for (uint32_t y = 1; y < rows - 1; y++) {
            for (uint32_t x = 1; x < cols - 1; x++) {
                if (temp[y * cols + x] == 0) continue; // Skip background pixels

                int p1 = (temp[y * cols + x] > 0);
                int p2 = (temp[(y - 1) * cols + x] > 0);
                int p3 = (temp[(y - 1) * cols + x + 1] > 0);
                int p4 = (temp[y * cols + x + 1] > 0);
                int p5 = (temp[(y + 1) * cols + x + 1] > 0);
                int p6 = (temp[(y + 1) * cols + x] > 0);
                int p7 = (temp[(y + 1) * cols + x - 1] > 0);
                int p8 = (temp[y * cols + x - 1] > 0);
                int p9 = (temp[(y - 1) * cols + x - 1] > 0);

                int a = 0;
                if (!p2 && p3) a++;
                if (!p3 && p4) a++;
                if (!p4 && p5) a++;
                if (!p5 && p6) a++;
                if (!p6 && p7) a++;
                if (!p7 && p8) a++;
                if (!p8 && p9) a++;
                if (!p9 && p2) a++;

                int b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

                if (b >= 2 && b <= 6 && a == 1 && (p2 * p4 * p6 == 0) && (p4 * p6 * p8 == 0)) {
                    result[y * cols + x] = 0; // Delete pixel
                    changed = 1;
                }
            }
        }

        memcpy(temp, result, rows * cols);

        // --- Second sub-iteration ---
        for (uint32_t y = 1; y < rows - 1; y++) {
            for (uint32_t x = 1; x < cols - 1; x++) {
                 if (temp[y * cols + x] == 0) continue; // Skip background pixels

                int p1 = (temp[y * cols + x] > 0);
                int p2 = (temp[(y - 1) * cols + x] > 0);
                int p3 = (temp[(y - 1) * cols + x + 1] > 0);
                int p4 = (temp[y * cols + x + 1] > 0);
                int p5 = (temp[(y + 1) * cols + x + 1] > 0);
                int p6 = (temp[(y + 1) * cols + x] > 0);
                int p7 = (temp[(y + 1) * cols + x - 1] > 0);
                int p8 = (temp[y * cols + x - 1] > 0);
                int p9 = (temp[(y - 1) * cols + x - 1] > 0);


                int a = 0;
                if (!p2 && p3) a++;
                if (!p3 && p4) a++;
                if (!p4 && p5) a++;
                if (!p5 && p6) a++;
                if (!p6 && p7) a++;
                if (!p7 && p8) a++;
                if (!p8 && p9) a++;
                if (!p9 && p2) a++;

                int b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

                if (b >= 2 && b <= 6 && a == 1 && (p2 * p4 * p8 == 0) && (p2 * p6 * p8 == 0)) {
                    result[y * cols + x] = 0;  // Delete pixel
                    changed = 1;
                }
            }
        }
        free(temp);
    } while (changed);
}


void preprocessImage(uint8_t *image, uint8_t *processedImage,
                    uint32_t rows, uint32_t cols,
                    PreprocessingOptions *options) {

    if (image == NULL || processedImage == NULL || options == NULL) {
        return; // Handle null pointers
    }

    // Create a temporary buffer to store intermediate results
    uint8_t *tempImage = (uint8_t *)malloc(rows * cols);
    if (tempImage == NULL) {
        return; // Memory allocation failed
    }

    // 1. Initial Copy: Copy input image to temporary buffer
    memcpy(tempImage, image, rows * cols);
    uint8_t * current = tempImage;
    uint8_t * next = processedImage;


    // 2. Adaptive Thresholding
    if (options->applyThresholding) {
        adaptiveThreshold(current, next, 15, 2, rows, cols); // windowSize=15, c=2 are common defaults
        //swap buffers so we don't need to malloc a new one each time.
        uint8_t* swap = current;
        current = next;
        next = swap;

    }

    // 3. Noise Removal
    if (options->applyNoiseRemoval) {
        removeNoise(current, next, options->noiseThreshold, rows, cols);
         //swap buffers
        uint8_t* swap = current;
        current = next;
        next = swap;
    }

     // 4.  Slant Correction
    if (options->applySlantCorrection) {
        int minX, minY, maxX, maxY;
        findBoundingBox(current, rows, cols, &minX, &minY, &maxX, &maxY);
         // Check if a valid bounding box was found before calculating slant.
        if (maxX > minX && maxY > minY) {
            double slantAngle = calculateSlant(current, minX, minY, maxX, maxY, rows, cols);
            //Limit the slant angle.
            if (fabs(slantAngle) > options->slantAngleLimit) {
                slantAngle = (slantAngle > 0) ? options->slantAngleLimit : -options->slantAngleLimit;
            }
            correctSlant(current, next, slantAngle, rows, cols);
              //swap buffers
            uint8_t* swap = current;
            current = next;
            next = swap;

        } else {
            // If no valid bounding box, just copy the current image to the next.
            memcpy(next, current, rows * cols);
        }
    }


    // 5. Size Normalization
    if (options->applyNormalization) {
        normalizeSize(current, next, options->borderSize, rows, cols);
         //swap buffers
        uint8_t* swap = current;
        current = next;
        next = swap;
    }

    // 6.  Stroke Width Normalization
    if (options->applyStrokeNorm) {
        normalizeStrokeWidth(current, next, options->targetStrokeWidth, rows, cols);
          //swap buffers
        uint8_t* swap = current;
        current = next;
        next = swap;
    }

    // 7. Thinning
    if (options->applyThinning) {
        thinImage(current, next, rows, cols);
          //swap buffers
        uint8_t* swap = current;
        current = next;
        next = swap;
    }

    //If no processing happened, next still equals processedImage
    //but current points to tempImage. So copy that.
    if(current != next){
        memcpy(processedImage, current, rows*cols);
    }

    free(tempImage); // Free the temporary buffer
}


void preprocessDataset(MNISTDataset *dataset, MNISTDataset *processedDataset,
                      PreprocessingOptions *options) {

    if (dataset == NULL || processedDataset == NULL || options == NULL) {
        return; // Handle null pointers
    }

    // Copy the dataset properties
    processedDataset->numImages = dataset->numImages;
    processedDataset->rows = dataset->rows;
    processedDataset->cols = dataset->cols;
    processedDataset->imageSize = dataset->imageSize;
    
    // Allocate memory for the processed dataset
    processedDataset->images = (uint8_t *)malloc(dataset->numImages * dataset->imageSize * sizeof(uint8_t));
    processedDataset->labels = (uint8_t *)malloc(dataset->numImages * sizeof(uint8_t));

    if (processedDataset->images == NULL || processedDataset->labels == NULL) {
        // Handle memory allocation failure
        fprintf(stderr, "Memory allocation failed for processed dataset.\n");
        // Clean up any allocated memory before returning
        if (processedDataset->images) free(processedDataset->images);
        if (processedDataset->labels) free(processedDataset->labels);
        processedDataset->images = NULL;
        processedDataset->labels = NULL;
        return;
    }
    
    // Copy all labels
    memcpy(processedDataset->labels, dataset->labels, dataset->numImages * sizeof(uint8_t));
    
    printf("Processing %u images...\n", dataset->numImages);
    
    // Process each image
    for (uint32_t i = 0; i < dataset->numImages; i++) {
        // Get pointers to the current image in each dataset
        uint8_t *srcImage = dataset->images + (i * dataset->imageSize);
        uint8_t *dstImage = processedDataset->images + (i * dataset->imageSize);
        
        // Apply preprocessing to this image
        preprocessImage(srcImage, dstImage, dataset->rows, dataset->cols, options);
        
        // Progress indicator (every 1000 images)
        if (i % 1000 == 0) {
            printf("Processed %u/%u images\r", i, dataset->numImages);
            fflush(stdout);
        }
    }
    
    printf("\nPreprocessing complete: %u images processed\n", dataset->numImages);
}