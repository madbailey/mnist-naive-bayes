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

// Original function to predict the digit for a single image (for backward compatibility)
uint8_t predictNaiveBayes(NaiveBayesModel *model, double *features) {
    PredictionResult result = predictNaiveBayesWithConfidence(model, features, 1);
    uint8_t prediction = result.prediction;
    freePredictionResult(&result);
    return prediction;
}

// Enhanced prediction function that returns confidence and top-N predictions
PredictionResult predictNaiveBayesWithConfidence(NaiveBayesModel *model, double *features, int topN) {
    PredictionResult result;
    
    // Ensure topN is within bounds
    if (topN > model->numClasses) {
        topN = model->numClasses;
    }
    if (topN < 1) {
        topN = 1;
    }
    
    // Allocate memory for class probabilities and top predictions
    result.classProbs = (double*)malloc(model->numClasses * sizeof(double));
    result.topN = (uint8_t*)malloc(topN * sizeof(uint8_t));
    result.n = topN;
    
    // Initialize arrays
    for (int c = 0; c < model->numClasses; c++) {
        result.classProbs[c] = 0.0;
    }
    
    for (int i = 0; i < topN; i++) {
        result.topN[i] = 0;
    }
    
    // Calculate log probability for each class
    double *logProbs = (double*)malloc(model->numClasses * sizeof(double));
    
    for (int c = 0; c < model->numClasses; c++) {
        double logProb = log(model->classPrior[c]);
        
        for (int f = 0; f < model->numFeatures; f++) {
            // Ensure feature value is in valid range
            double featureVal = features[f];
            featureVal = (featureVal < 0) ? 0 : (featureVal > 1.0 ? 1.0 : featureVal);
            
            // Determine which bin the orientation falls into
            int bin = (int)(featureVal / model->binWidth);
            
            // Safety check for valid bin index
            bin = (bin < 0) ? 0 : (bin >= model->numBins ? model->numBins - 1 : bin);
            
            // Add log probability from this feature
            double prob = model->featureProb[c][f][bin];
            
            // Ensure probability is not zero (avoid log(0))
            prob = (prob < 1e-10) ? 1e-10 : prob;
            
            logProb += log(prob);
        }
        
        logProbs[c] = logProb;
    }
    
    // Convert log probabilities to actual probabilities (normalized)
    double maxLogProb = -INFINITY;
    for (int c = 0; c < model->numClasses; c++) {
        if (logProbs[c] > maxLogProb) {
            maxLogProb = logProbs[c];
        }
    }
    
    double sumProb = 0.0;
    for (int c = 0; c < model->numClasses; c++) {
        // Subtract maxLogProb to avoid numerical instability
        result.classProbs[c] = exp(logProbs[c] - maxLogProb);
        sumProb += result.classProbs[c];
    }
    
    // Normalize
    for (int c = 0; c < model->numClasses; c++) {
        result.classProbs[c] /= sumProb;
    }
    
    // Find top N predictions
    for (int i = 0; i < topN; i++) {
        double maxProb = -1;
        int maxIdx = -1;
        
        for (int c = 0; c < model->numClasses; c++) {
            if (result.classProbs[c] > maxProb) {
                maxProb = result.classProbs[c];
                maxIdx = c;
            }
        }
        
        if (maxIdx >= 0) {
            result.topN[i] = (uint8_t)maxIdx;
            result.classProbs[maxIdx] = -1;  // Mark as used
        }
    }
    
    // Restore class probabilities (undo the marking)
    for (int c = 0; c < model->numClasses; c++) {
        if (result.classProbs[c] < 0) {
            // Find the position in topN to determine the rank
            for (int i = 0; i < topN; i++) {
                if (result.topN[i] == c) {
                    // Restore probability based on the normalized log probability
                    result.classProbs[c] = exp(logProbs[c] - maxLogProb) / sumProb;
                    break;
                }
            }
        }
    }
    
    // Set the prediction to the top class
    result.prediction = result.topN[0];
    
    // Calculate confidence as the difference between top two probabilities
    double topProb = result.classProbs[result.topN[0]];
    double runnerUpProb = (topN > 1) ? result.classProbs[result.topN[1]] : 0.0;
    
    // Confidence can be calculated in different ways:
    // 1. Simple probability: result.confidence = topProb;
    // 2. Margin between top two: result.confidence = topProb - runnerUpProb;
    // 3. Normalized margin: result.confidence = (topProb - runnerUpProb) / topProb;
    
    // We'll use the simple probability approach for better interpretability
    result.confidence = topProb;
    
    // Clean up temporary array
    free(logProbs);
    
    return result;
}

// Free prediction result resources
void freePredictionResult(PredictionResult *result) {
    if (result->classProbs) {
        free(result->classProbs);
        result->classProbs = NULL;
    }
    
    if (result->topN) {
        free(result->topN);
        result->topN = NULL;
    }
    
    result->n = 0;
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