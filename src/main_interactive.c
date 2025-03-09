#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "mnist_loader.h"
#include "hog.h"
#include "naive_bayes.h"
#include "utils.h"
#include "ui_drawer.h"
#include "normalization.h"
#include "feature_selection.h"
#include "specialized_classifier.h"

// Function to adjust dataset labels to be 0-based
void adjustLabels(MNISTDataset *dataset) {
    printf("Adjusting labels to be 0-based...\n");
    for (uint32_t i = 0; i < dataset->numImages; i++) {
        if (dataset->labels[i] > 0) {
            dataset->labels[i] -= 1;  // Make 1-26 into 0-25
        }
    }
}

int main(int argc, char *argv[]) {
    // Determine if we're recognizing digits or letters
    int recognizeLetters = 1;  // Default to letters

    // Check command line arguments
    if (argc > 1) {
        if (strcmp(argv[1], "digits") == 0) {
            recognizeLetters = 0;
        } else if (strcmp(argv[1], "letters") == 0) {
            recognizeLetters = 1;
        } else {
            printf("Usage: %s [digits|letters]\n", argv[0]);
            return 1;
        }
    }

    MNISTDataset trainDataset;
    MNISTDataset processedTrainDataset; // Add a dataset for the processed images
    HOGFeatures trainHOG;
    NaiveBayesModel model;
    SpecializedClassifierManager specializedManager;

    // IMPORTANT: These parameters must match those in ui_drawer.c
    int cellSize = 4;  // MAKE SURE THIS MATCHES THE CELL_SIZE IN ui_drawer.c
    int numBins = 9;
    int numClasses = recognizeLetters ? 26 : 10;  // 26 for letters, 10 for digits

    // Set file paths based on what we're recognizing
    const char *imageFile, *labelFile;
    if (recognizeLetters) {
        imageFile = "data/emnist-letters-train-images-idx3-ubyte";
        labelFile = "data/emnist-letters-train-labels-idx1-ubyte";
        printf("Running letter recognizer\n");
    } else {
        imageFile = "data/train-images-idx3-ubyte";
        labelFile = "data/train-labels-idx1-ubyte";
        printf("Running digit recognizer\n");
    }

    // Load training data
    printf("Loading training data...\n");
    if (recognizeLetters) {
        // Use EMNIST-specific loader for letters
        if (!loadEMNISTDataset(imageFile, labelFile, &trainDataset)) {
            printf("Failed to load training data. Check that files exist in the data/ directory.\n");
            return 1;
        }
    } else {
        // Use standard loader for digits
        if (!loadMNISTDataset(imageFile, labelFile, &trainDataset)) {
            printf("Failed to load training data. Check that files exist in the data/ directory.\n");
            return 1;
        }
    }
    printf("Loaded %u training images\n", trainDataset.numImages);

    // Adjust labels for letters (they're 1-indexed in EMNIST)
    if (recognizeLetters) {
        adjustLabels(&trainDataset);
    }

    // --- Preprocessing ---
    PreprocessingOptions options;
    initDefaultPreprocessing(&options);
    
    // Configure preprocessing options for the interactive application
    // These settings are optimized for training data and can be adjusted
    options.applyNormalization = 1;     // Center and scale the image
    options.applyThresholding = 1;      // Apply adaptive thresholding
    options.applySlantCorrection = 1;   // Correct slant angle for consistent characters
    options.applyNoiseRemoval = 1;      // Remove small noise artifacts
    options.applyStrokeNorm = 1;        // Normalize stroke width for consistent features
    options.applyThinning = 0;          // Thinning can sometimes reduce accuracy
    
    // Fine-tune parameters
    options.borderSize = 2;             // Small border around normalized character
    options.targetStrokeWidth = 2;      // Target stroke width after normalization
    options.noiseThreshold = 2;         // Remove noise specks of this size or smaller
    options.slantAngleLimit = 0.4;      // Maximum slant angle to correct (in radians)

    printf("Preprocessing dataset with custom options...\n");
    printf("- Normalization: %s\n", options.applyNormalization ? "ON" : "OFF");
    printf("- Thresholding: %s\n", options.applyThresholding ? "ON" : "OFF");
    printf("- Slant correction: %s\n", options.applySlantCorrection ? "ON" : "OFF");
    printf("- Noise removal: %s\n", options.applyNoiseRemoval ? "ON" : "OFF");
    printf("- Stroke normalization: %s\n", options.applyStrokeNorm ? "ON" : "OFF");
    printf("- Thinning: %s\n", options.applyThinning ? "ON" : "OFF");
    
    preprocessDataset(&trainDataset, &processedTrainDataset, &options);
    printf("Preprocessing complete.\n");


    // Initialize HOG feature structure
    trainHOG.numImages = processedTrainDataset.numImages; // Use the processed dataset
    trainHOG.numFeatures = (processedTrainDataset.rows / cellSize) * (processedTrainDataset.cols / cellSize) * numBins;

    // Extract HOG features *from the processed dataset*
    printf("Extracting HOG features...\n");
    extractHOGFeatures(&processedTrainDataset, &trainHOG, cellSize, numBins); // Pass processed dataset

    // Apply feature selection to reduce dimensionality and noise
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
    
    // Create reduced feature set from the selected features
    HOGFeatures reducedTrainHOG;
    reducedTrainHOG.numImages = trainHOG.numImages;
    reducedTrainHOG.numFeatures = numSelected;
    
    printf("Creating reduced feature set...\n");
    if (!createReducedFeatureSet(&trainHOG, &reducedTrainHOG, selectedIndices, numSelected)) {
        printf("Failed to create reduced feature set\n");
        free(selectedIndices);
        return 1;
    }
    
    printf("Successfully reduced feature dimensionality from %u to %u\n", 
           trainHOG.numFeatures, numSelected);

    // Initialize and train the model with reduced features
    printf("\n===== Training Model with Selected Features =====\n");
    if (!initNaiveBayes(&model, numClasses, reducedTrainHOG.numFeatures, numBins, 1.0)) {
        printf("Failed to initialize Naive Bayes model\n");
        free(selectedIndices);
        return 1;
    }

    trainNaiveBayes(&model, &reducedTrainHOG);
    printf("Model trained and ready!\n");
    
    // Initialize specialized classifier manager
    printf("\n===== Setting Up Specialized Classifiers =====\n");
    const int MAX_SPECIALIZED_CLASSIFIERS = 5;
    
    if (!initSpecializedClassifierManager(&specializedManager, MAX_SPECIALIZED_CLASSIFIERS)) {
        printf("Failed to initialize specialized classifier manager\n");
        return 1;
    }
    
    // Define commonly confused letter pairs for specialized classifiers
    // We'll train specialized classifiers for these pairs
    typedef struct {
        uint8_t class1;
        uint8_t class2;
        double threshold;
    } ConfusedPair;
    
    // Initialize with known confused pairs (only for letter recognition)
    if (recognizeLetters) {
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
                    numBins,      // numBins
                    1.0)) {       // alpha
                printf("Failed to add specialized classifier for pair %d\n", i);
                continue;
            }
            
            // Train this specialized classifier
            if (!trainSpecializedClassifier(&specializedManager, i, &reducedTrainHOG)) {
                printf("Failed to train specialized classifier for pair %d\n", i);
                continue;
            }
        }
    }
    
    // Store selected feature indices in a file for future use
    FILE *featureIdxFile = fopen("selected_features.dat", "wb");
    if (featureIdxFile) {
        fwrite(&numSelected, sizeof(uint32_t), 1, featureIdxFile);
        fwrite(selectedIndices, sizeof(uint32_t), numSelected, featureIdxFile);
        fclose(featureIdxFile);
        printf("Saved selected feature indices to selected_features.dat\n");
    }

    // Load reference samples for visualization
    printf("Loading reference samples for visualization...\n");
    if (!loadReferenceSamples(imageFile, labelFile)) {  // Note:  Loads the *original*, unprocessed samples
        printf("Warning: Failed to load reference samples. Visualization will be limited.\n");
    }

    // Initialize drawing UI
    DrawingUI ui;
    if (!initUI(&ui, &model, &specializedManager, numClasses, recognizeLetters)) {
        printf("Failed to initialize UI\n");
        return 1;
    }

    // Main loop
    int running = 1;
    while (running) {
        running = processEvents(&ui);
        renderUI(&ui);

        // Cap to ~60 FPS
        SDL_Delay(16);
    }

    // Clean up
    cleanupUI(&ui);
    freeMNISTDataset(&trainDataset);
    freeMNISTDataset(&processedTrainDataset); // Free the processed dataset
    freeHOGFeatures(&trainHOG);
    freeHOGFeatures(&reducedTrainHOG);
    freeNaiveBayes(&model);
    freeSpecializedClassifierManager(&specializedManager);
    free(selectedIndices);

    return 0;
}