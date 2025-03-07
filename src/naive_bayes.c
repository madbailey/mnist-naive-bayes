#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "naive_bayes.h"
#include "mnist_loader.h"

//basic naive bayes implementation 

void initNaiveBayes(NaiveBayesModel *model, int numBins, double alpha) {
    model->numBins = numBins;
    model->binWidth = 256 / numBins;
    model->alpha = alpha;

    //set probs to zero to start
    memset(model->pixelProb, 0, sizeof(model->pixelProb));
    memset(model->classPrior, 0, sizeof(model->classPrior));
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