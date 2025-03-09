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

// Structure to hold prediction results
typedef struct {
    uint8_t prediction;         // The predicted class
    double confidence;          // Confidence score (0-1)
    double *classProbs;         // Array of probabilities for each class
    uint8_t *topN;              // Array containing indices of top N predictions
    int n;                      // Number of top predictions stored
} PredictionResult;

// Function to initialize the Naive Bayes model
bool initNaiveBayes(NaiveBayesModel *model, int numClasses, int numFeatures, int numBins, double alpha);

// Function to train the Naive Bayes model
void trainNaiveBayes(NaiveBayesModel *model, HOGFeatures *hogFeatures);

// Original prediction function (for backward compatibility)
uint8_t predictNaiveBayes(NaiveBayesModel *model, double *features);

// Enhanced prediction function that returns confidence and top-N predictions
PredictionResult predictNaiveBayesWithConfidence(NaiveBayesModel *model, double *features, int topN);

// Free prediction result resources
void freePredictionResult(PredictionResult *result);

// Free Naive Bayes model resources
void freeNaiveBayes(NaiveBayesModel *model);


#endif // NAIVE_BAYES_H