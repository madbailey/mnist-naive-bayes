#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "naive_bayes.h"
#include "mnist_loader.h"

//basic naive bayes implementation 

bool initNaiveBayes(NaiveBayesModel *model, int numBins, double alpha, uint32_t imageSize) {
    model->numBins = numBins;
    model->binWidth = 256 / numBins;
    model->alpha = alpha;
    model->imageSize = imageSize;

    model->pixelProb = malloc(10 * sizeof(double **));
    if (!model->pixelProb)
        return false;

    for (uint32_t i = 0; i < 10; i++) {
        model->pixelProb[i] = malloc(imageSize * sizeof(double *));
        if (!model->pixelProb[i]) {
            // Free previously allocated pointers to avoid memory leak
            for (uint32_t j = 0; j < i; j++) {
                free(model->pixelProb[j]);
            }
            free(model->pixelProb);
            return false;
        }
        for (uint32_t j = 0; j < imageSize; j++) {
            model->pixelProb[i][j] = calloc(numBins, sizeof(double));
            if (!model->pixelProb[i][j]) {
                // Free memory allocated for current row
                for (uint32_t k = 0; k < j; k++) {
                    free(model->pixelProb[i][k]);
                }
                free(model->pixelProb[i]);
                // Free memory for previous rows
                for (uint32_t k = 0; k < i; k++) {
                    for (uint32_t l = 0; l < imageSize; l++) {
                        free(model->pixelProb[k][l]);
                    }
                    free(model->pixelProb[k]);
                }
                free(model->pixelProb);
                return false;
            }
        }
    }

    // Initialize class prior probabilities to zero
    memset(model->classPrior, 0, sizeof(model->classPrior));
    return true;
}

//map pixel intensity to bin
int getBin(int intensity, int binWidth) {
    return intensity /binWidth;
}

void trainNaiveBayes(NaiveBayesModel *model, MNISTDataset *dataset) {
    int counts[10][784][256] = {0};
    int classCounts[10] = {0};
    
    // Go through all training images
    for (uint32_t i = 0; i < dataset->numImages; i++) {
        uint8_t label = dataset->labels[i];
        classCounts[label]++;
        
        // Count pixel intensities
        for (uint32_t j = 0; j < dataset->imageSize; j++) {
            uint8_t pixel = dataset->images[i * dataset->imageSize + j];
            int bin = getBin(pixel, model->binWidth);
            counts[label][j][bin]++;
        }
    }
    
    // Calculate class priors
    for (int c = 0; c < 10; c++) {
        model->classPrior[c] = (double)classCounts[c] / dataset->numImages;
    }
    
    // Calculate pixel probabilities with Laplace smoothing
    for (int c = 0; c < 10; c++) {
        for (uint32_t j = 0; j < dataset->imageSize; j++) {
            for (int b = 0; b < model->numBins; b++) {
                // Apply Laplace smoothing
                model->pixelProb[c][j][b] = 
                    (counts[c][j][b] + model->alpha) / 
                    (classCounts[c] + model->alpha * model->numBins);
            }
        }
    }
    
}
void freeNaiveBayes(NaiveBayesModel *model) {
    for (uint32_t i = 0; i < 10; i++) {
        for (uint32_t j = 0; j < model->imageSize; j++) {
            free(model->pixelProb[i][j]);
        }
        free(model->pixelProb[i]);
    }
    free(model->pixelProb);
    model->pixelProb = NULL;
}
// Function to predict the digit for a single image
uint8_t predictNaiveBayes(NaiveBayesModel *model, uint8_t *image, uint32_t imageSize) {
    double logProb[10] = {0};
    
    // Calculate log probability for each class
    for (int c = 0; c < 10; c++) {
        // Start with log of class prior
        logProb[c] = log(model->classPrior[c]);
        
        // Add log probabilities of each pixel
        for (uint32_t j = 0; j < imageSize; j++) {
            int bin = getBin(image[j], model->binWidth);
            logProb[c] += log(model->pixelProb[c][j][bin]);
        }
    }
    
    // Find the class with the highest log probability
    uint8_t bestClass = 0;
    double bestLogProb = logProb[0];
    
    for (int c = 1; c < 10; c++) {
        if (logProb[c] > bestLogProb) {
            bestLogProb = logProb[c];
            bestClass = c;
        }
    }
    
    return bestClass;
}