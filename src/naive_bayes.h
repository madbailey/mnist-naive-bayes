#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <stdint.h>
#include "mnist_loader.h"
#include <stdbool.h>
typedef struct {
    // Dynamic 3D array for probabilities
    double ***pixelProb;  // [class][pixel_position][intensity_bin]
    // Prior probability of each class
    double classPrior[10];
    // Number of intensity bins
    int numBins;
    // Bin width
    int binWidth;
    // Laplace smoothing parameter
    double alpha;
    // Image size
    uint32_t imageSize;
} NaiveBayesModel;

// Function to initialize the Naive Bayes model
bool initNaiveBayes(NaiveBayesModel *model, int numBins, double alpha, uint32_t imageSize);

// Function to map a pixel intensity to a bin
int getBin(int intensity, int binWidth);

// Function to train the Naive Bayes model
void trainNaiveBayes(NaiveBayesModel *model, MNISTDataset *dataset);

void freeNaiveBayes(NaiveBayesModel *model);

// Function to predict the digit for a single image
uint8_t predictNaiveBayes(NaiveBayesModel *model, uint8_t *image, uint32_t imageSize);

#endif // NAIVE_BAYES_H