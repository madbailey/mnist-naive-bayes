#ifndef FEATURE_SELECTION_H
#define FEATURE_SELECTION_H

#include "hog.h"
#include <stdint.h>
#include <stdbool.h>

// Structure to represent a feature's importance score
typedef struct {
    uint32_t index;    // Original feature index
    double score;      // Importance score
} FeatureScore;

// Feature selection methods
#define FS_METHOD_VARIANCE      0  // Select features with highest variance
#define FS_METHOD_CHI_SQUARE    1  // Chi-square test for feature relevance
#define FS_METHOD_MUTUAL_INFO   2  // Mutual information criterion

/**
 * Select the most discriminative features from a dataset
 * 
 * @param hogFeatures Input HOG features
 * @param numClasses Number of classes in the dataset
 * @param selectedIndices Array where selected feature indices will be stored
 * @param numToSelect Number of features to select
 * @param method Selection method (FS_METHOD_*)
 * @return Number of features actually selected
 */
uint32_t selectDiscriminativeFeatures(
    HOGFeatures *hogFeatures, 
    int numClasses,
    uint32_t *selectedIndices, 
    uint32_t numToSelect,
    int method
);

/**
 * Select features that best discriminate between specific classes
 * 
 * @param hogFeatures Input HOG features
 * @param targetClasses Array of class indices to focus discrimination on
 * @param numTargetClasses Number of target classes
 * @param selectedIndices Array where selected feature indices will be stored
 * @param numToSelect Number of features to select
 * @param method Selection method (FS_METHOD_*)
 * @return Number of features actually selected
 */
uint32_t selectClassSpecificFeatures(
    HOGFeatures *hogFeatures,
    uint8_t *targetClasses,
    uint32_t numTargetClasses,
    uint32_t *selectedIndices,
    uint32_t numToSelect,
    int method
);

/**
 * Create a reduced feature set containing only selected features
 * 
 * @param hogFeatures Input HOG features
 * @param reducedFeatures Output reduced HOG features (must be pre-allocated)
 * @param selectedIndices Array of selected feature indices
 * @param numSelected Number of selected features
 * @return true if successful, false otherwise
 */
bool createReducedFeatureSet(
    HOGFeatures *hogFeatures,
    HOGFeatures *reducedFeatures,
    uint32_t *selectedIndices,
    uint32_t numSelected
);

#endif // FEATURE_SELECTION_H