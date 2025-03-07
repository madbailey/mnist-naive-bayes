#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <stdint.h>
#include "mnist_loader.h"

// Structure to hold our Naive Bayes model
typedef struct {
    // Probability of each pixel intensity bin for each class (0-9) and pixel position
    double pixelProb[10][784][256];  // [class][pixel_position][intensity_bin]
    // Prior probability of each class
    double classPrior[10];
    // Number of intensity bins
    int numBins;
    // Bin width
    int binWidth;
    // Laplace smoothing parameter
    double alpha;
} NaiveBayesModel;

// Function to initialize the Naive Bayes model
void initNaiveBayes(NaiveBayesModel *model, int numBins, double alpha);

// Function to map a pixel intensity to a bin
int getBin(int intensity, int binWidth);

// Function to train the Naive Bayes model
void trainNaiveBayes(NaiveBayesModel *model, MNISTDataset *dataset);

// Function to predict the digit for a single image
uint8_t predictNaiveBayes(NaiveBayesModel *model, uint8_t *image, uint32_t imageSize);

#endif // NAIVE_BAYES_H