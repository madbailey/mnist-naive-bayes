#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "naive_bayes.h"
#include "mnist_loader.h"

// naive bayes implementation with hog
static int getHOGBin(double value, double binWidth) {
    int bin = (int)(value/ binWidth);
    return (bin >=0) ? (bin < 256 ? bin : 255) : 0;
}

bool initNaiveBayes(NaiveBayesModel *model, int numClasses, int numFeatures, int numBins, double alpha) {
    model->numClasses = numClasses;
    model->numFeatures = numFeatures;
    model->numBins = numBins;
    model->binWidth = 1.0 / numBins;
    model->alpha = alpha;

    model->classPrior = (double*)malloc(numClasses * sizeof(double));
    if (model->classPrior == NULL) {
        printf("Failed to allocate memory for class priors\n");
        return false;
    }

    // Allocate memory for feature probabilities
    model->featureProb = (double***)malloc(numClasses * sizeof(double**));
    if (model->featureProb == NULL) {
        free(model->classPrior);
        printf("Failed to allocate memory for feature probabilities\n");
        return false;
    }

    for (int c = 0; c < numClasses; c++) {
        model->featureProb[c] = (double**)malloc(numFeatures * sizeof(double*));
        if (model->featureProb[c] == NULL) {
            // Clean up previously allocated memory
            for (int i = 0; i < c; i++) {
                free(model->featureProb[i]);
            }
            free(model->featureProb);
            free(model->classPrior);
            printf("Failed to allocate memory for feature probabilities\n");
            return false;
        }

        for (int f = 0; f < numFeatures; f++) {
            model->featureProb[c][f] = (double*)malloc(numBins * sizeof(double));
            if (model->featureProb[c][f] == NULL) {
                // Clean up previously allocated memory
                for (int j = 0; j < f; j++) {
                    free(model->featureProb[c][j]);
                }
                for (int i = 0; i < c; i++) {
                    for (int j = 0; j < numFeatures; j++) {
                        free(model->featureProb[i][j]);
                    }
                    free(model->featureProb[i]);
                }
                free(model->featureProb);
                free(model->classPrior);
                printf("Failed to allocate memory for feature probabilities\n");
                return false;
            }
            memset(model->featureProb[c][f], 0, numBins * sizeof(double));
        }
    }
    
    memset(model->classPrior, 0, numClasses * sizeof(double));
    
    printf("Initialized HOG Naive Bayes model with %d classes, %d features, %d bins\n", 
           numClasses, numFeatures, numBins);
    
    return true;
}
void trainNaiveBayes(NaiveBayesModel *model, HOGFeatures *hogFeatures) {
    if (model->numFeatures != hogFeatures->numFeatures) {
        printf("Error: Feature count mismatch\n");
        return;
    }

    int ***counts;
    int *classCounts;

    // Allocate memory for counts
    counts = (int***)malloc(model->numClasses * sizeof(int**));
    for (int c = 0; c < model->numClasses; c++) {
        counts[c] = (int**)malloc(model->numFeatures * sizeof(int*));
        for (int f = 0; f < model->numFeatures; f++) {
            counts[c][f] = (int*)malloc(model->numBins * sizeof(int));
            memset(counts[c][f], 0, model->numBins * sizeof(int));
        }
    }

    classCounts = (int*)malloc(model->numClasses * sizeof(int));
    memset(classCounts, 0, model->numClasses * sizeof(int));

    // Count feature occurrences
    for (uint32_t i = 0; i < hogFeatures->numImages; i++) {
        uint8_t label = hogFeatures->labels[i];
        if (label >= model->numClasses) {
            printf("Warning: Label %d out of range\n", label);
            continue;
        }
        
        classCounts[label]++;
        
        double *features = &hogFeatures->features[i * hogFeatures->numFeatures];
        for (int f = 0; f < model->numFeatures; f++) {
            int bin = getHOGBin(features[f], model->binWidth);
            counts[label][f][bin]++;
        }
    }

    // Calculate class priors
    for (int c = 0; c < model->numClasses; c++) {
        model->classPrior[c] = (double)classCounts[c] / hogFeatures->numImages;
    }

    // Calculate feature probabilities with Laplace smoothing
    for (int c = 0; c < model->numClasses; c++) {
        for (int f = 0; f < model->numFeatures; f++) {
            for (int b = 0; b < model->numBins; b++) {
                model->featureProb[c][f][b] = 
                    (counts[c][f][b] + model->alpha) / 
                    (classCounts[c] + model->alpha * model->numBins);
            }
        }
    }

    // Free temporary memory
    for (int c = 0; c < model->numClasses; c++) {
        for (int f = 0; f < model->numFeatures; f++) {
            free(counts[c][f]);
        }
        free(counts[c]);
    }
    free(counts);
    free(classCounts);

    printf("Trained HOG Naive Bayes model\n");
}

// Function to predict the digit for a single image
uint8_t predictNaiveBayes(NaiveBayesModel *model, double *features) {
    double maxLogProb = -INFINITY;
    int bestClass = 0;
    
    // Calculate log probability for each class
    for (int c = 0; c < model->numClasses; c++) {
        double logProb = log(model->classPrior[c]);
        
        for (int f = 0; f < model->numFeatures; f++) {
            int bin = getHOGBin(features[f], model->binWidth);
            logProb += log(model->featureProb[c][f][bin]);
        }
        
        if (logProb > maxLogProb) {
            maxLogProb = logProb;
            bestClass = c;
        }
    }
    
    return (uint8_t)bestClass;
}

void freeNaiveBayes(NaiveBayesModel *model) {
    for (int c = 0; c < model->numClasses; c++) {
        for (int f = 0; f < model->numFeatures; f++) {
            free(model->featureProb[c][f]);
        }
        free(model->featureProb[c]);
    }
    free(model->featureProb);
    free(model->classPrior);
}