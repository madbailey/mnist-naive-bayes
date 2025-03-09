#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mnist_loader.h"
#include "hog.h"
#include "naive_bayes.h"
#include "utils.h"
#include "normalization.h"
#include "feature_selection.h"
#include "specialized_classifier.h"

// Function to convert numeric label to character
char labelToChar(uint8_t label) {
    // EMNIST letters use 1=A, 2=B, ..., 26=Z
    return 'A' + (label - 1);
}

// Preprocess the dataset to make labels 0-based 
void adjustLabels(MNISTDataset *dataset) {
    printf("Adjusting labels to be 0-based...\n");
    for (uint32_t i = 0; i < dataset->numImages; i++) {
        if (dataset->labels[i] > 0) {
            dataset->labels[i] -= 1;  // Make 1-26 into 0-25
        } else {
            printf("Warning: Unexpected label value %d\n", dataset->labels[i]);
        }
    }
}

int main() {
    MNISTDataset trainDataset, testDataset;
    MNISTDataset processedTrainDataset, processedTestDataset; // New preprocessed datasets
    HOGFeatures trainHOG, testHOG;
    NaiveBayesModel model;
    PreprocessingOptions preOptions; // Preprocessing options
    
    int cellSize = 4;
    int numBins = 9;
    int numClasses = 26; // 26 letters (A-Z)

    // Load training data
    printf("Loading EMNIST letter training data...\n");
    if (!loadMNISTDataset("data/emnist-letters-train-images-idx3-ubyte", 
                          "data/emnist-letters-train-labels-idx1-ubyte", 
                          &trainDataset)) {
        printf("Failed to load training data. Check that files exist in the data/ directory.\n");
        return 1;
    }
    printf("Loaded %u training letter images\n", trainDataset.numImages);
    
    // Load test data
    printf("Loading EMNIST letter test data...\n");
    if (!loadMNISTDataset("data/emnist-letters-test-images-idx3-ubyte", 
                          "data/emnist-letters-test-labels-idx1-ubyte", 
                          &testDataset)) {
        printf("Failed to load test data. Check that files exist in the data/ directory.\n");
        freeMNISTDataset(&trainDataset);
        return 1;
    }
    printf("Loaded %u test letter images\n", testDataset.numImages);

    // Adjust labels to be 0-based for our model
    adjustLabels(&trainDataset);
    adjustLabels(&testDataset);

    // Initialize preprocessing options with defaults
    initDefaultPreprocessing(&preOptions);
    
    // Preprocess training data
    printf("Preprocessing training data...\n");
    preprocessDataset(&trainDataset, &processedTrainDataset, &preOptions);
    
    // Preprocess test data
    printf("Preprocessing test data...\n");
    preprocessDataset(&testDataset, &processedTestDataset, &preOptions);
    
    // Debug: print some stats about the preprocessed datasets
    printf("Preprocessed training set: %u images, %ux%u size\n", 
           processedTrainDataset.numImages, 
           processedTrainDataset.rows, 
           processedTrainDataset.cols);
    printf("Preprocessed test set: %u images, %ux%u size\n", 
           processedTestDataset.numImages, 
           processedTestDataset.rows, 
           processedTestDataset.cols);

    // Initialize HOG feature structures for preprocessed data
    trainHOG.numImages = processedTrainDataset.numImages;
    trainHOG.numFeatures = (processedTrainDataset.rows/cellSize) * (processedTrainDataset.cols/cellSize) * numBins;
    
    testHOG.numImages = processedTestDataset.numImages;
    testHOG.numFeatures = (processedTestDataset.rows/cellSize) * (processedTestDataset.cols/cellSize) * numBins;

    // Extract HOG features from preprocessed data
    printf("Extracting HOG features from preprocessed training letters...\n");
    extractHOGFeatures(&processedTrainDataset, &trainHOG, cellSize, numBins);

    printf("Extracting HOG features from preprocessed test letters...\n");
    extractHOGFeatures(&processedTestDataset, &testHOG, cellSize, numBins);

    // Apply feature selection to reduce dimensionality
    printf("\n===== Feature Selection =====\n");
    
    // Define % of features to select (e.g., 40% of original features)
    float featureSelectionRatio = 0.4;
    uint32_t numToSelect = (uint32_t)(trainHOG.numFeatures * featureSelectionRatio);
    printf("Selecting %u features (%.1f%% of original %u features)\n", 
           numToSelect, featureSelectionRatio * 100, trainHOG.numFeatures);
    
    // Allocate memory for selected feature indices
    uint32_t *selectedIndices = (uint32_t *)malloc(numToSelect * sizeof(uint32_t));
    if (!selectedIndices) {
        printf("Failed to allocate memory for feature selection\n");
        return 1;
    }
    
    // Implement hybrid feature selection strategy:
    // 1. First, focus on most confused letter pairs (i and l)
    // 2. Then add general discriminative features
    
    // Define most confused letter pairs (i and l in 0-indexed values)
    // 'i' is the 8th letter (0-indexed), 'l' is the 11th letter
    uint8_t confusedPairsForFeatureSelection[] = {8, 11}; // i (8) and l (11)
    uint32_t numConfusedClasses = sizeof(confusedPairsForFeatureSelection) / sizeof(confusedPairsForFeatureSelection[0]);
    
    // Determine how many features to select for specific letter pairs (40% of total)
    uint32_t numPairSpecificFeatures = numToSelect * 0.4;
    
    printf("\n----- Class-Specific Feature Selection -----\n");
    printf("Selecting %u features specific to discriminating between confused letter pairs (i, l)\n", 
           numPairSpecificFeatures);
    
    // Allocate memory for class-specific features
    uint32_t *pairSpecificIndices = (uint32_t *)malloc(numPairSpecificFeatures * sizeof(uint32_t));
    if (!pairSpecificIndices) {
        printf("Failed to allocate memory for class-specific feature selection\n");
        free(selectedIndices);
        return 1;
    }
    
    // Select features specific to confused letter pairs
    uint32_t numPairSelected = selectClassSpecificFeatures(
        &trainHOG, confusedPairsForFeatureSelection, numConfusedClasses, 
        pairSpecificIndices, numPairSpecificFeatures, FS_METHOD_CHI_SQUARE);
    
    if (numPairSelected == 0) {
        printf("Class-specific feature selection failed\n");
        free(pairSpecificIndices);
        free(selectedIndices);
        return 1;
    }
    
    // Now select general discriminative features for the remaining slots
    uint32_t numGeneralFeatures = numToSelect - numPairSelected;
    
    printf("\n----- General Feature Selection -----\n");
    printf("Selecting %u general discriminative features\n", numGeneralFeatures);
    
    // Allocate memory for general features
    uint32_t *generalIndices = (uint32_t *)malloc(numGeneralFeatures * sizeof(uint32_t));
    if (!generalIndices) {
        printf("Failed to allocate memory for general feature selection\n");
        free(pairSpecificIndices);
        free(selectedIndices);
        return 1;
    }
    
    // Select general discriminative features
    uint32_t numGeneralSelected = selectDiscriminativeFeatures(
        &trainHOG, numClasses, generalIndices, numGeneralFeatures, FS_METHOD_CHI_SQUARE);
    
    if (numGeneralSelected == 0) {
        printf("General feature selection failed\n");
        free(generalIndices);
        free(pairSpecificIndices);
        free(selectedIndices);
        return 1;
    }
    
    // Combine class-specific and general features
    printf("\n----- Combining Feature Sets -----\n");
    printf("Combining %u class-specific and %u general features\n", 
           numPairSelected, numGeneralSelected);
    
    // Copy class-specific features first
    for (uint32_t i = 0; i < numPairSelected; i++) {
        selectedIndices[i] = pairSpecificIndices[i];
    }
    
    // Copy general features
    for (uint32_t i = 0; i < numGeneralSelected; i++) {
        selectedIndices[numPairSelected + i] = generalIndices[i];
    }
    
    uint32_t numSelected = numPairSelected + numGeneralSelected;
    
    // Clean up temporary arrays
    free(pairSpecificIndices);
    free(generalIndices);
    
    if (numSelected == 0) {
        printf("Feature selection failed\n");
        free(selectedIndices);
        return 1;
    }
    
    // Create reduced feature sets
    HOGFeatures reducedTrainHOG, reducedTestHOG;
    reducedTrainHOG.numImages = trainHOG.numImages;
    reducedTrainHOG.numFeatures = numSelected;
    
    reducedTestHOG.numImages = testHOG.numImages;
    reducedTestHOG.numFeatures = numSelected;
    
    printf("Creating reduced feature sets...\n");
    if (!createReducedFeatureSet(&trainHOG, &reducedTrainHOG, selectedIndices, numSelected) ||
        !createReducedFeatureSet(&testHOG, &reducedTestHOG, selectedIndices, numSelected)) {
        printf("Failed to create reduced feature sets\n");
        free(selectedIndices);
        return 1;
    }
    
    printf("Successfully reduced feature dimensionality from %u to %u\n", 
           trainHOG.numFeatures, numSelected);
    
    // Save selected feature indices for later use
    FILE *featureIdxFile = fopen("selected_features.dat", "wb");
    if (featureIdxFile) {
        fwrite(&numSelected, sizeof(uint32_t), 1, featureIdxFile);
        fwrite(selectedIndices, sizeof(uint32_t), numSelected, featureIdxFile);
        fclose(featureIdxFile);
        printf("Saved selected feature indices to selected_features.dat\n");
    }

    // Initialize and train the model with reduced features
    printf("\n===== Training Model with Selected Features =====\n");
    if (!initNaiveBayes(&model, numClasses, reducedTrainHOG.numFeatures, 32, 1.0)) {
        printf("Failed to initialize Naive Bayes model\n");
        free(selectedIndices);
        return 1;
    }
    
    trainNaiveBayes(&model, &reducedTrainHOG);
    
    // Initialize specialized classifier manager
    printf("\n===== Setting Up Specialized Classifiers =====\n");
    SpecializedClassifierManager specializedManager;
    const int MAX_SPECIALIZED_CLASSIFIERS = 5;
    
    if (!initSpecializedClassifierManager(&specializedManager, MAX_SPECIALIZED_CLASSIFIERS)) {
        printf("Failed to initialize specialized classifier manager\n");
        return 1;
    }
    
    // Get commonly confused letter pairs from the analysis
    // We'll start with 'i' and 'l' which are index 8 and 11 in 0-based indexing
    // We'll add more based on the confusion matrix results
    
    // Define confused pairs we want to focus on
    typedef struct {
        uint8_t class1;
        uint8_t class2;
        double threshold;
    } ConfusedPair;
    
    // Initialize with known confused pairs
    ConfusedPair confusedPairs[] = {
        {8, 11, 0.7},  // 'i' and 'l'
        {14, 20, 0.7}, // 'o' and 'u'
        {2, 6, 0.7}    // 'c' and 'g'
    };
    int numConfusedPairs = sizeof(confusedPairs) / sizeof(confusedPairs[0]);
    
    // Add specialized classifiers for each pair
    for (int i = 0; i < numConfusedPairs; i++) {
        printf("Setting up specialized classifier for letters %c and %c...\n",
               'a' + confusedPairs[i].class1, 'a' + confusedPairs[i].class2);
        
        if (!addSpecializedClassifier(
                &specializedManager,
                confusedPairs[i].class1,
                confusedPairs[i].class2,
                confusedPairs[i].threshold,
                reducedTrainHOG.numFeatures,
                32,   // numBins (same as general classifier)
                1.0)) // alpha (same as general classifier)
        {
            printf("Failed to add specialized classifier for pair %d\n", i);
            continue;
        }
        
        // Train this specialized classifier
        if (!trainSpecializedClassifier(&specializedManager, i, &reducedTrainHOG)) {
            printf("Failed to train specialized classifier for pair %d\n", i);
            continue;
        }
    }
    
    // Test the model with two-stage classification
    printf("\n===== Testing Two-Stage Classification =====\n");
    int correct = 0;
    int specializedCorrect = 0;
    int specializedTotal = 0;
    int confusionMatrix[26][26] = {0}; // Track misclassifications
    
    for (uint32_t i = 0; i < reducedTestHOG.numImages; i++) {
        double *features = &reducedTestHOG.features[i * reducedTestHOG.numFeatures];
        
        // Get both the general prediction and the two-stage prediction
        uint8_t generalPrediction = predictNaiveBayes(&model, features);
        PredictionResult twoStageResult = twoStageClassify(&model, &specializedManager, features, 3);
        
        uint8_t actual = reducedTestHOG.labels[i];
        
        // Update confusion matrix (now with 0-based labels)
        if (actual < 26 && twoStageResult.prediction < 26) {
            confusionMatrix[actual][twoStageResult.prediction]++;
        }
        
        // Check if the specialized classifier was used for this sample
        bool usedSpecialized = (generalPrediction != twoStageResult.prediction);
        
        if (usedSpecialized) {
            specializedTotal++;
            if (twoStageResult.prediction == actual) {
                specializedCorrect++;
            }
        }
        
        if (twoStageResult.prediction == actual) {
            correct++;
        }
        
        // Free prediction result resources
        freePredictionResult(&twoStageResult);
        
        // Print progress
        if ((i + 1) % 1000 == 0 || i + 1 == testHOG.numImages) {
            printf("Progress: %u/%u, Accuracy: %.2f%%\n", 
                   i + 1, testHOG.numImages, 
                   100.0 * correct / (i + 1));
        }
    }
    
    // Final accuracy
    double accuracy = 100.0 * correct / testHOG.numImages;
    printf("Final accuracy: %.2f%%\n", accuracy);
    
    // Specialized classifier stats
    if (specializedTotal > 0) {
        double specializedAccuracy = 100.0 * specializedCorrect / specializedTotal;
        printf("\nSpecialized classifiers were used for %d samples (%.2f%% of test set)\n", 
               specializedTotal, 100.0 * specializedTotal / testHOG.numImages);
        printf("Accuracy of specialized classifiers: %.2f%%\n", specializedAccuracy);
    } else {
        printf("\nNo specialized classifiers were used in testing\n");
    }
    
    // Print most confused letter pairs
    printf("\nTop letter confusions:\n");
    printf("Actual\tPredicted\tCount\n");
    printf("------\t---------\t-----\n");
    
    // Find top confusions (naive approach)
    for (int n = 0; n < 10; n++) { // Show top 10 confusions
        int maxCount = 0;
        int maxActual = 0;
        int maxPredicted = 0;
        
        for (int i = 0; i < 26; i++) {
            for (int j = 0; j < 26; j++) {
                if (i != j && confusionMatrix[i][j] > maxCount) {
                    maxCount = confusionMatrix[i][j];
                    maxActual = i;
                    maxPredicted = j;
                }
            }
        }
        
        if (maxCount > 0) {
            printf("%c\t%c\t\t%d\n", 
                   labelToChar(maxActual+1), 
                   labelToChar(maxPredicted+1), 
                   maxCount);
            
            // Zero out this entry so we find the next highest
            confusionMatrix[maxActual][maxPredicted] = 0;
        }
    }
    
    // Print per-letter accuracies
    printf("\nPer-letter accuracy:\n");
    printf("Letter\tAccuracy\n");
    printf("------\t--------\n");
    
    for (int i = 0; i < 26; i++) {
        int totalLetters = 0;
        for (int j = 0; j < 26; j++) {
            totalLetters += confusionMatrix[i][j];
        }
        
        double letterAccuracy = totalLetters > 0 ? 
            100.0 * confusionMatrix[i][i] / totalLetters : 0.0;
            
        printf("%c\t%.2f%%\n", labelToChar(i+1), letterAccuracy);
    }
    
    // Free memory for both original and preprocessed datasets
    freeMNISTDataset(&trainDataset);
    freeMNISTDataset(&testDataset);
    freeMNISTDataset(&processedTrainDataset);
    freeMNISTDataset(&processedTestDataset);
    freeHOGFeatures(&trainHOG);
    freeHOGFeatures(&testHOG);
    freeHOGFeatures(&reducedTrainHOG);
    freeHOGFeatures(&reducedTestHOG);
    freeNaiveBayes(&model);
    freeSpecializedClassifierManager(&specializedManager);
    free(selectedIndices);
    
    return 0;
}