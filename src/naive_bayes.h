#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include "hog.h"
#include <stdint.h>
#include "mnist_loader.h"
#include <stdbool.h>
typedef struct {
    
    int numClasses;
    uint32_t numFeatures;
    int numBins;
    double binWidth;
    double alpha;

    double ***featureProb;

    double *classPrior;
} NaiveBayesModel;

// Function to initialize the Naive Bayes model
bool initNaiveBayes(NaiveBayesModel *model, int numClasses, int numFeatures, int numBins, double alpha);

// Function to train the Naive Bayes model
void trainNaiveBayes(NaiveBayesModel *model, HOGFeatures *hogFeatures);

uint8_t predictNaiveBayes(NaiveBayesModel *model, double *features);

void freeNaiveBayes(NaiveBayesModel *model);



#endif // NAIVE_BAYES_H