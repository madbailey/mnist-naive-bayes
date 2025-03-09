#ifndef SPECIALIZED_CLASSIFIER_H
#define SPECIALIZED_CLASSIFIER_H

#include "naive_bayes.h"
#include "hog.h"
#include <stdint.h>
#include <stdbool.h>

// Structure to represent a specialized classifier for a commonly confused pair
typedef struct {
    uint8_t class1;          // First class in the pair (e.g., 'i')
    uint8_t class2;          // Second class in the pair (e.g., 'l')
    NaiveBayesModel model;   // Specialized model for just these two classes
    double confidenceThreshold; // Threshold for applying this classifier
} SpecializedClassifier;

// Structure to manage multiple specialized classifiers
typedef struct {
    SpecializedClassifier *classifiers; // Array of specialized classifiers
    int numClassifiers;                 // Number of classifiers in the array
} SpecializedClassifierManager;

// Initialize the specialized classifier manager
bool initSpecializedClassifierManager(SpecializedClassifierManager *manager, int maxClassifiers);

// Add a specialized classifier for a commonly confused pair
bool addSpecializedClassifier(
    SpecializedClassifierManager *manager,
    uint8_t class1, 
    uint8_t class2,
    double confidenceThreshold,
    int numFeatures,
    int numBins,
    double alpha
);

// Train a specialized classifier with filtered training data
bool trainSpecializedClassifier(
    SpecializedClassifierManager *manager,
    int classifierIndex,
    HOGFeatures *hogFeatures
);

// Apply the two-stage classification process
PredictionResult twoStageClassify(
    NaiveBayesModel *generalModel,
    SpecializedClassifierManager *manager,
    double *features,
    int topN
);

// Free memory allocated for specialized classifiers
void freeSpecializedClassifierManager(SpecializedClassifierManager *manager);

#endif // SPECIALIZED_CLASSIFIER_H