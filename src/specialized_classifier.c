#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include "specialized_classifier.h"
#include "naive_bayes.h"
#include "hog.h"

// Initialize the specialized classifier manager
bool initSpecializedClassifierManager(SpecializedClassifierManager *manager, int maxClassifiers) {
    manager->classifiers = (SpecializedClassifier *)malloc(maxClassifiers * sizeof(SpecializedClassifier));
    if (!manager->classifiers) {
        printf("Failed to allocate memory for specialized classifiers\n");
        return false;
    }
    
    manager->numClassifiers = 0;
    return true;
}

// Add a specialized classifier for a commonly confused pair
bool addSpecializedClassifier(
    SpecializedClassifierManager *manager,
    uint8_t class1, 
    uint8_t class2,
    double confidenceThreshold,
    int numFeatures,
    int numBins,
    double alpha
) {
    int index = manager->numClassifiers;
    SpecializedClassifier *classifier = &manager->classifiers[index];
    
    classifier->class1 = class1;
    classifier->class2 = class2;
    classifier->confidenceThreshold = confidenceThreshold;
    
    // Initialize a specialized naive Bayes model with just 2 classes
    if (!initNaiveBayes(&classifier->model, 2, numFeatures, numBins, alpha)) {
        printf("Failed to initialize specialized classifier for classes %d and %d\n", class1, class2);
        return false;
    }
    
    manager->numClassifiers++;
    printf("Added specialized classifier for classes %d and %d (letters %c and %c)\n", 
           class1, class2, 'A' + class1, 'A' + class2);
    
    return true;
}

// Create a filtered training set containing only the two classes of interest
static HOGFeatures* createFilteredTrainingSet(HOGFeatures *origFeatures, uint8_t class1, uint8_t class2) {
    // Count how many samples we have of the two classes
    uint32_t count = 0;
    for (uint32_t i = 0; i < origFeatures->numImages; i++) {
        if (origFeatures->labels[i] == class1 || origFeatures->labels[i] == class2) {
            count++;
        }
    }
    
    if (count == 0) {
        printf("Warning: No training samples found for classes %d and %d\n", class1, class2);
        return NULL;
    }
    
    // Allocate a new HOGFeatures structure for the filtered data
    HOGFeatures *filtered = (HOGFeatures *)malloc(sizeof(HOGFeatures));
    if (!filtered) {
        printf("Failed to allocate memory for filtered features\n");
        return NULL;
    }
    
    filtered->numImages = count;
    filtered->numFeatures = origFeatures->numFeatures;
    
    // Allocate memory for features and labels
    filtered->features = (double *)malloc(count * filtered->numFeatures * sizeof(double));
    filtered->labels = (uint8_t *)malloc(count * sizeof(uint8_t));
    
    if (!filtered->features || !filtered->labels) {
        printf("Failed to allocate memory for filtered features data\n");
        if (filtered->features) free(filtered->features);
        if (filtered->labels) free(filtered->labels);
        free(filtered);
        return NULL;
    }
    
    // Copy the filtered samples
    uint32_t filteredIdx = 0;
    for (uint32_t i = 0; i < origFeatures->numImages; i++) {
        if (origFeatures->labels[i] == class1 || origFeatures->labels[i] == class2) {
            // Copy features
            memcpy(&filtered->features[filteredIdx * filtered->numFeatures],
                   &origFeatures->features[i * origFeatures->numFeatures],
                   filtered->numFeatures * sizeof(double));
            
            // Set label to 0 for class1 and 1 for class2 (binary classification)
            filtered->labels[filteredIdx] = (origFeatures->labels[i] == class1) ? 0 : 1;
            
            filteredIdx++;
        }
    }
    
    printf("Created filtered training set with %u samples for classes %d and %d\n", 
           count, class1, class2);
    
    return filtered;
}

// Train a specialized classifier with filtered training data
bool trainSpecializedClassifier(
    SpecializedClassifierManager *manager,
    int classifierIndex,
    HOGFeatures *hogFeatures
) {
    if (classifierIndex >= manager->numClassifiers) {
        printf("Invalid classifier index\n");
        return false;
    }
    
    SpecializedClassifier *classifier = &manager->classifiers[classifierIndex];
    
    // Create a filtered training set with only the two classes of interest
    HOGFeatures *filteredFeatures = createFilteredTrainingSet(
        hogFeatures, classifier->class1, classifier->class2);
    
    if (!filteredFeatures) {
        return false;
    }
    
    // Train the specialized classifier on the filtered dataset
    trainNaiveBayes(&classifier->model, filteredFeatures);
    
    printf("Trained specialized classifier for classes %d and %d\n", 
           classifier->class1, classifier->class2);
    
    // Free the filtered training set
    free(filteredFeatures->features);
    free(filteredFeatures->labels);
    free(filteredFeatures);
    
    return true;
}

// Find the appropriate specialized classifier for a prediction
static int findSpecializedClassifier(
    SpecializedClassifierManager *manager,
    uint8_t class1,
    uint8_t class2
) {
    for (int i = 0; i < manager->numClassifiers; i++) {
        SpecializedClassifier *classifier = &manager->classifiers[i];
        
        // Check if this classifier handles the given class pair (in any order)
        if ((classifier->class1 == class1 && classifier->class2 == class2) ||
            (classifier->class1 == class2 && classifier->class2 == class1)) {
            return i;
        }
    }
    
    return -1; // No matching classifier found
}

// Apply the two-stage classification process
PredictionResult twoStageClassify(
    NaiveBayesModel *generalModel,
    SpecializedClassifierManager *manager,
    double *features,
    int topN
) {
    // First stage: general classification with confidence
    PredictionResult result = predictNaiveBayesWithConfidence(generalModel, features, topN);
    
    // If we have good confidence, just return the result
    if (result.confidence > 0.8) { // High confidence threshold
        return result;
    }
    
    // If we have at least 2 predictions and the top two are close
    if (result.n >= 2) {
        uint8_t topClass = result.topN[0];
        uint8_t secondClass = result.topN[1];
        
        // Check if we have a specialized classifier for these two classes
        int classifierIdx = findSpecializedClassifier(manager, topClass, secondClass);
        
        if (classifierIdx >= 0) {
            SpecializedClassifier *classifier = &manager->classifiers[classifierIdx];
            
            // Check if confidence is below the threshold for applying specialized classifier
            if (result.confidence < classifier->confidenceThreshold) {
                // Apply the specialized classifier
                PredictionResult specializedResult = predictNaiveBayesWithConfidence(&classifier->model, features, 2);
                
                // Map the binary result back to the original classes
                uint8_t specializedPrediction;
                if (specializedResult.prediction == 0) {
                    specializedPrediction = classifier->class1;
                } else {
                    specializedPrediction = classifier->class2;
                }
                
                // If the specialized classifier is confident, use its prediction
                if (specializedResult.confidence > 0.7) {
                    // Update the primary result with the specialized classifier's decision
                    // but keep all the class probabilities the same
                    uint8_t originalPrediction = result.prediction;
                    result.prediction = specializedPrediction;
                    
                    // Swap the positions in the topN array
                    for (int i = 0; i < result.n; i++) {
                        if (result.topN[i] == specializedPrediction) {
                            result.topN[i] = originalPrediction;
                            result.topN[0] = specializedPrediction;
                            break;
                        }
                    }
                    
                    printf("Specialized classifier overrode prediction from %d to %d\n", 
                           originalPrediction, specializedPrediction);
                }
                
                // Free the specialized result resources
                freePredictionResult(&specializedResult);
            }
        }
    }
    
    return result;
}

// Free memory allocated for specialized classifiers
void freeSpecializedClassifierManager(SpecializedClassifierManager *manager) {
    if (manager->classifiers) {
        for (int i = 0; i < manager->numClassifiers; i++) {
            freeNaiveBayes(&manager->classifiers[i].model);
        }
        free(manager->classifiers);
        manager->classifiers = NULL;
    }
    manager->numClassifiers = 0;
}