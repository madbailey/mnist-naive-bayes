#include "feature_selection.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Comparison function for sorting FeatureScore structures in descending order
static int compareFeatureScores(const void *a, const void *b) {
    const FeatureScore *fa = (const FeatureScore *)a;
    const FeatureScore *fb = (const FeatureScore *)b;
    
    // Sort in descending order by score
    if (fa->score > fb->score) return -1;
    if (fa->score < fb->score) return 1;
    return 0;
}

// Calculate variance of a feature across all samples
static double calculateVariance(double *values, uint32_t numSamples) {
    if (numSamples <= 1) return 0.0;
    
    // Calculate mean
    double sum = 0.0;
    for (uint32_t i = 0; i < numSamples; i++) {
        sum += values[i];
    }
    double mean = sum / numSamples;
    
    // Calculate variance
    double variance = 0.0;
    for (uint32_t i = 0; i < numSamples; i++) {
        double diff = values[i] - mean;
        variance += diff * diff;
    }
    
    return variance / (numSamples - 1);
}

// Calculate chi-square statistic for a feature
static double calculateChiSquare(double *values, uint8_t *labels, 
                                uint32_t numSamples, int numClasses) {
    if (numSamples <= 1) return 0.0;
    
    // Count samples in each class
    uint32_t *classCounts = (uint32_t *)calloc(numClasses, sizeof(uint32_t));
    if (!classCounts) return 0.0;
    
    // To simplify, we'll discretize continuous features into bins
    const int numBins = 8;
    
    // Count observations in each bin for each class
    uint32_t **binClassCounts = (uint32_t **)malloc(numBins * sizeof(uint32_t *));
    if (!binClassCounts) {
        free(classCounts);
        return 0.0;
    }
    
    for (int b = 0; b < numBins; b++) {
        binClassCounts[b] = (uint32_t *)calloc(numClasses, sizeof(uint32_t));
        if (!binClassCounts[b]) {
            for (int i = 0; i < b; i++) {
                free(binClassCounts[i]);
            }
            free(binClassCounts);
            free(classCounts);
            return 0.0;
        }
    }
    
    // Find min and max values for binning
    double minVal = values[0];
    double maxVal = values[0];
    for (uint32_t i = 1; i < numSamples; i++) {
        if (values[i] < minVal) minVal = values[i];
        if (values[i] > maxVal) maxVal = values[i];
    }
    double binWidth = (maxVal - minVal) / numBins;
    if (binWidth <= 0.0) binWidth = 1.0; // Handle case where all values are identical
    
    // Count samples in each bin for each class
    for (uint32_t i = 0; i < numSamples; i++) {
        uint8_t label = labels[i];
        if (label >= numClasses) continue; // Skip invalid labels
        
        classCounts[label]++;
        
        // Determine which bin this value falls into
        int bin = (int)((values[i] - minVal) / binWidth);
        if (bin >= numBins) bin = numBins - 1; // Handle edge case
        if (bin < 0) bin = 0; // Handle edge case
        
        binClassCounts[bin][label]++;
    }
    
    // Calculate chi-square statistic
    double chiSquare = 0.0;
    for (int b = 0; b < numBins; b++) {
        uint32_t binTotal = 0;
        for (int c = 0; c < numClasses; c++) {
            binTotal += binClassCounts[b][c];
        }
        
        for (int c = 0; c < numClasses; c++) {
            if (classCounts[c] == 0) continue; // Skip empty classes
            
            // Calculate expected count
            double expected = (double)binTotal * classCounts[c] / numSamples;
            if (expected < 1e-10) continue; // Avoid division by zero
            
            // Add to chi-square
            double observed = binClassCounts[b][c];
            double diff = observed - expected;
            chiSquare += (diff * diff) / expected;
        }
    }
    
    // Clean up
    for (int b = 0; b < numBins; b++) {
        free(binClassCounts[b]);
    }
    free(binClassCounts);
    free(classCounts);
    
    return chiSquare;
}

// Calculate mutual information between a feature and class labels
static double calculateMutualInformation(double *values, uint8_t *labels, 
                                       uint32_t numSamples, int numClasses) {
    if (numSamples <= 1) return 0.0;
    
    // To simplify, we'll discretize continuous features into bins
    const int numBins = 8;
    
    // Count samples in each class and bin
    uint32_t *classCounts = (uint32_t *)calloc(numClasses, sizeof(uint32_t));
    uint32_t *binCounts = (uint32_t *)calloc(numBins, sizeof(uint32_t));
    uint32_t **jointCounts = (uint32_t **)malloc(numBins * sizeof(uint32_t *));
    
    if (!classCounts || !binCounts || !jointCounts) {
        if (classCounts) free(classCounts);
        if (binCounts) free(binCounts);
        if (jointCounts) free(jointCounts);
        return 0.0;
    }
    
    for (int b = 0; b < numBins; b++) {
        jointCounts[b] = (uint32_t *)calloc(numClasses, sizeof(uint32_t));
        if (!jointCounts[b]) {
            for (int i = 0; i < b; i++) {
                free(jointCounts[i]);
            }
            free(jointCounts);
            free(classCounts);
            free(binCounts);
            return 0.0;
        }
    }
    
    // Find min and max values for binning
    double minVal = values[0];
    double maxVal = values[0];
    for (uint32_t i = 1; i < numSamples; i++) {
        if (values[i] < minVal) minVal = values[i];
        if (values[i] > maxVal) maxVal = values[i];
    }
    double binWidth = (maxVal - minVal) / numBins;
    if (binWidth <= 0.0) binWidth = 1.0; // Handle case where all values are identical
    
    // Count samples in each bin and class
    for (uint32_t i = 0; i < numSamples; i++) {
        uint8_t label = labels[i];
        if (label >= numClasses) continue; // Skip invalid labels
        
        classCounts[label]++;
        
        // Determine which bin this value falls into
        int bin = (int)((values[i] - minVal) / binWidth);
        if (bin >= numBins) bin = numBins - 1; // Handle edge case
        if (bin < 0) bin = 0; // Handle edge case
        
        binCounts[bin]++;
        jointCounts[bin][label]++;
    }
    
    // Calculate mutual information
    double mutualInfo = 0.0;
    for (int b = 0; b < numBins; b++) {
        if (binCounts[b] == 0) continue;
        
        double pBin = (double)binCounts[b] / numSamples;
        
        for (int c = 0; c < numClasses; c++) {
            if (classCounts[c] == 0 || jointCounts[b][c] == 0) continue;
            
            double pClass = (double)classCounts[c] / numSamples;
            double pJoint = (double)jointCounts[b][c] / numSamples;
            
            // MI += p(bin,class) * log( p(bin,class) / (p(bin) * p(class)) )
            mutualInfo += pJoint * log(pJoint / (pBin * pClass));
        }
    }
    
    // Clean up
    for (int b = 0; b < numBins; b++) {
        free(jointCounts[b]);
    }
    free(jointCounts);
    free(binCounts);
    free(classCounts);
    
    return mutualInfo;
}

uint32_t selectDiscriminativeFeatures(
    HOGFeatures *hogFeatures, 
    int numClasses,
    uint32_t *selectedIndices, 
    uint32_t numToSelect,
    int method) {
    
    // Validate input
    if (!hogFeatures || !selectedIndices || hogFeatures->numImages == 0 || 
        hogFeatures->numFeatures == 0 || numToSelect == 0) {
        return 0;
    }
    
    // Cap selection to the available features
    if (numToSelect > hogFeatures->numFeatures) {
        numToSelect = hogFeatures->numFeatures;
    }
    
    // Allocate array for feature scores
    FeatureScore *scores = (FeatureScore *)malloc(hogFeatures->numFeatures * sizeof(FeatureScore));
    if (!scores) {
        printf("Failed to allocate memory for feature scores\n");
        return 0;
    }
    
    // Initialize scores
    for (uint32_t f = 0; f < hogFeatures->numFeatures; f++) {
        scores[f].index = f;
        scores[f].score = 0.0;
    }
    
    // Calculate feature scores based on selected method
    printf("Calculating feature scores using method %d...\n", method);
    
    switch (method) {
        case FS_METHOD_VARIANCE: {
            // Allocate temporary array for feature values
            double *values = (double *)malloc(hogFeatures->numImages * sizeof(double));
            if (!values) {
                printf("Failed to allocate memory for feature values\n");
                free(scores);
                return 0;
            }
            
            // Calculate variance for each feature
            for (uint32_t f = 0; f < hogFeatures->numFeatures; f++) {
                // Extract values for this feature from all samples
                for (uint32_t i = 0; i < hogFeatures->numImages; i++) {
                    values[i] = hogFeatures->features[i * hogFeatures->numFeatures + f];
                }
                
                // Calculate variance
                scores[f].score = calculateVariance(values, hogFeatures->numImages);
                
                // Progress indicator
                if ((f + 1) % 100 == 0 || f + 1 == hogFeatures->numFeatures) {
                    printf("\rEvaluated %u/%u features...", f + 1, hogFeatures->numFeatures);
                    fflush(stdout);
                }
            }
            
            free(values);
            printf("\nCompleted variance calculations\n");
            break;
        }
        
        case FS_METHOD_CHI_SQUARE: {
            // Allocate temporary array for feature values
            double *values = (double *)malloc(hogFeatures->numImages * sizeof(double));
            if (!values) {
                printf("Failed to allocate memory for feature values\n");
                free(scores);
                return 0;
            }
            
            // Calculate chi-square for each feature
            for (uint32_t f = 0; f < hogFeatures->numFeatures; f++) {
                // Extract values for this feature from all samples
                for (uint32_t i = 0; i < hogFeatures->numImages; i++) {
                    values[i] = hogFeatures->features[i * hogFeatures->numFeatures + f];
                }
                
                // Calculate chi-square
                scores[f].score = calculateChiSquare(values, hogFeatures->labels, 
                                                   hogFeatures->numImages, numClasses);
                
                // Progress indicator
                if ((f + 1) % 100 == 0 || f + 1 == hogFeatures->numFeatures) {
                    printf("\rEvaluated %u/%u features...", f + 1, hogFeatures->numFeatures);
                    fflush(stdout);
                }
            }
            
            free(values);
            printf("\nCompleted chi-square calculations\n");
            break;
        }
        
        case FS_METHOD_MUTUAL_INFO: {
            // Allocate temporary array for feature values
            double *values = (double *)malloc(hogFeatures->numImages * sizeof(double));
            if (!values) {
                printf("Failed to allocate memory for feature values\n");
                free(scores);
                return 0;
            }
            
            // Calculate mutual information for each feature
            for (uint32_t f = 0; f < hogFeatures->numFeatures; f++) {
                // Extract values for this feature from all samples
                for (uint32_t i = 0; i < hogFeatures->numImages; i++) {
                    values[i] = hogFeatures->features[i * hogFeatures->numFeatures + f];
                }
                
                // Calculate mutual information
                scores[f].score = calculateMutualInformation(values, hogFeatures->labels, 
                                                          hogFeatures->numImages, numClasses);
                
                // Progress indicator
                if ((f + 1) % 100 == 0 || f + 1 == hogFeatures->numFeatures) {
                    printf("\rEvaluated %u/%u features...", f + 1, hogFeatures->numFeatures);
                    fflush(stdout);
                }
            }
            
            free(values);
            printf("\nCompleted mutual information calculations\n");
            break;
        }
        
        default:
            printf("Unknown feature selection method: %d\n", method);
            free(scores);
            return 0;
    }
    
    // Sort features by score in descending order
    qsort(scores, hogFeatures->numFeatures, sizeof(FeatureScore), compareFeatureScores);
    
    // Select top features
    for (uint32_t i = 0; i < numToSelect; i++) {
        selectedIndices[i] = scores[i].index;
    }
    
    // Print score range for selected features
    printf("Selected %u features with scores ranging from %.6f to %.6f\n", 
           numToSelect, scores[0].score, scores[numToSelect-1].score);
    
    free(scores);
    return numToSelect;
}

bool createReducedFeatureSet(
    HOGFeatures *hogFeatures,
    HOGFeatures *reducedFeatures,
    uint32_t *selectedIndices,
    uint32_t numSelected) {
    
    // Validate input
    if (!hogFeatures || !reducedFeatures || !selectedIndices || 
        hogFeatures->numImages == 0 || numSelected == 0) {
        return false;
    }
    
    // Initialize reduced feature structure
    reducedFeatures->numImages = hogFeatures->numImages;
    reducedFeatures->numFeatures = numSelected;
    
    // Allocate memory for reduced features
    reducedFeatures->features = (double *)malloc(
        reducedFeatures->numImages * reducedFeatures->numFeatures * sizeof(double));
    
    if (!reducedFeatures->features) {
        printf("Failed to allocate memory for reduced feature set\n");
        return false;
    }
    
    // Copy labels if available
    if (hogFeatures->labels) {
        reducedFeatures->labels = (uint8_t *)malloc(reducedFeatures->numImages * sizeof(uint8_t));
        if (!reducedFeatures->labels) {
            printf("Failed to allocate memory for reduced feature labels\n");
            free(reducedFeatures->features);
            reducedFeatures->features = NULL;
            return false;
        }
        memcpy(reducedFeatures->labels, hogFeatures->labels, 
               reducedFeatures->numImages * sizeof(uint8_t));
    } else {
        reducedFeatures->labels = NULL;
    }
    
    // Copy selected features for each image
    for (uint32_t i = 0; i < hogFeatures->numImages; i++) {
        double *srcFeatures = &hogFeatures->features[i * hogFeatures->numFeatures];
        double *dstFeatures = &reducedFeatures->features[i * numSelected];
        
        for (uint32_t f = 0; f < numSelected; f++) {
            dstFeatures[f] = srcFeatures[selectedIndices[f]];
        }
        
        // Progress indicator
        if ((i + 1) % 10000 == 0 || i + 1 == hogFeatures->numImages) {
            printf("\rReduced feature set for %u/%u images...", 
                   i + 1, hogFeatures->numImages);
            fflush(stdout);
        }
    }
    
    printf("\nCreated reduced feature set with %u features per image\n", numSelected);
    return true;
}

// Calculate discrimination score between two specific classes for a feature
static double calculateClassDiscrimination(double *values, uint8_t *labels,
                                          uint32_t numSamples, uint8_t class1, uint8_t class2) {
    if (numSamples <= 1) return 0.0;
    
    // Collect values for each class
    double *class1Values = (double *)malloc(numSamples * sizeof(double));
    double *class2Values = (double *)malloc(numSamples * sizeof(double));
    uint32_t class1Count = 0;
    uint32_t class2Count = 0;
    
    if (!class1Values || !class2Values) {
        if (class1Values) free(class1Values);
        if (class2Values) free(class2Values);
        return 0.0;
    }
    
    // Separate values by class
    for (uint32_t i = 0; i < numSamples; i++) {
        if (labels[i] == class1) {
            class1Values[class1Count++] = values[i];
        } else if (labels[i] == class2) {
            class2Values[class2Count++] = values[i];
        }
    }
    
    // If either class has no samples, return 0
    if (class1Count == 0 || class2Count == 0) {
        free(class1Values);
        free(class2Values);
        return 0.0;
    }
    
    // Calculate means
    double class1Mean = 0.0;
    double class2Mean = 0.0;
    
    for (uint32_t i = 0; i < class1Count; i++) {
        class1Mean += class1Values[i];
    }
    class1Mean /= class1Count;
    
    for (uint32_t i = 0; i < class2Count; i++) {
        class2Mean += class2Values[i];
    }
    class2Mean /= class2Count;
    
    // Calculate variances
    double class1Var = 0.0;
    double class2Var = 0.0;
    
    for (uint32_t i = 0; i < class1Count; i++) {
        double diff = class1Values[i] - class1Mean;
        class1Var += diff * diff;
    }
    class1Var = (class1Count > 1) ? class1Var / (class1Count - 1) : 0.0;
    
    for (uint32_t i = 0; i < class2Count; i++) {
        double diff = class2Values[i] - class2Mean;
        class2Var += diff * diff;
    }
    class2Var = (class2Count > 1) ? class2Var / (class2Count - 1) : 0.0;
    
    // Calculate fisher score: (mean1 - mean2)^2 / (var1 + var2)
    double meanDiff = class1Mean - class2Mean;
    double sumVar = class1Var + class2Var;
    
    // Avoid division by zero
    double fisherScore = (sumVar > 1e-10) ? (meanDiff * meanDiff) / sumVar : 0.0;
    
    free(class1Values);
    free(class2Values);
    
    return fisherScore;
}

uint32_t selectClassSpecificFeatures(
    HOGFeatures *hogFeatures,
    uint8_t *targetClasses,
    uint32_t numTargetClasses,
    uint32_t *selectedIndices,
    uint32_t numToSelect,
    int method) {
    
    // Validate input
    if (!hogFeatures || !targetClasses || !selectedIndices || 
        hogFeatures->numImages == 0 || hogFeatures->numFeatures == 0 || 
        numTargetClasses < 2 || numToSelect == 0) {
        return 0;
    }
    
    // Cap selection to the available features
    if (numToSelect > hogFeatures->numFeatures) {
        numToSelect = hogFeatures->numFeatures;
    }
    
    // Print target classes we're focusing on
    printf("Targeting class-specific features for classes: ");
    for (uint32_t i = 0; i < numTargetClasses; i++) {
        if (i > 0) printf(", ");
        printf("%d", targetClasses[i]);
    }
    printf("\n");
    
    // Allocate array for feature scores
    FeatureScore *scores = (FeatureScore *)malloc(hogFeatures->numFeatures * sizeof(FeatureScore));
    if (!scores) {
        printf("Failed to allocate memory for feature scores\n");
        return 0;
    }
    
    // Initialize scores
    for (uint32_t f = 0; f < hogFeatures->numFeatures; f++) {
        scores[f].index = f;
        scores[f].score = 0.0;
    }
    
    // Allocate temporary array for feature values
    double *values = (double *)malloc(hogFeatures->numImages * sizeof(double));
    if (!values) {
        printf("Failed to allocate memory for feature values\n");
        free(scores);
        return 0;
    }
    
    // For each feature, calculate discrimination between all pairs of target classes
    for (uint32_t f = 0; f < hogFeatures->numFeatures; f++) {
        // Extract values for this feature from all samples
        for (uint32_t i = 0; i < hogFeatures->numImages; i++) {
            values[i] = hogFeatures->features[i * hogFeatures->numFeatures + f];
        }
        
        // Calculate average discrimination score across all class pairs
        double totalScore = 0.0;
        int numPairs = 0;
        
        // For each pair of target classes
        for (uint32_t i = 0; i < numTargetClasses; i++) {
            for (uint32_t j = i + 1; j < numTargetClasses; j++) {
                double pairScore;
                
                switch (method) {
                    case FS_METHOD_VARIANCE:
                    case FS_METHOD_CHI_SQUARE:
                    case FS_METHOD_MUTUAL_INFO:
                    default:
                        // For all methods, use Fisher score which is good for binary discrimination
                        pairScore = calculateClassDiscrimination(
                            values, hogFeatures->labels, 
                            hogFeatures->numImages, 
                            targetClasses[i], targetClasses[j]);
                        break;
                }
                
                totalScore += pairScore;
                numPairs++;
            }
        }
        
        // Calculate average score
        scores[f].score = (numPairs > 0) ? totalScore / numPairs : 0.0;
        
        // Progress indicator
        if ((f + 1) % 100 == 0 || f + 1 == hogFeatures->numFeatures) {
            printf("\rEvaluated %u/%u features...", f + 1, hogFeatures->numFeatures);
            fflush(stdout);
        }
    }
    
    printf("\nCompleted class-specific feature evaluation\n");
    
    // Sort features by score in descending order
    qsort(scores, hogFeatures->numFeatures, sizeof(FeatureScore), compareFeatureScores);
    
    // Select top features
    for (uint32_t i = 0; i < numToSelect; i++) {
        selectedIndices[i] = scores[i].index;
    }
    
    // Print score range for selected features
    printf("Selected %u class-specific features with scores ranging from %.6f to %.6f\n", 
           numToSelect, scores[0].score, scores[numToSelect-1].score);
    
    free(scores);
    free(values);
    return numToSelect;
}