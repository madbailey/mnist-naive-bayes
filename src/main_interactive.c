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

    // Initialize and train the model
    printf("Training model (this might take a minute)...\n");
    if (!initNaiveBayes(&model, numClasses, trainHOG.numFeatures, numBins, 1.0)) {
        printf("Failed to initialize Naive Bayes model\n");
        return 1;
    }

    trainNaiveBayes(&model, &trainHOG);
    printf("Model trained and ready!\n");

    // Load reference samples for visualization
    printf("Loading reference samples for visualization...\n");
    if (!loadReferenceSamples(imageFile, labelFile)) {  // Note:  Loads the *original*, unprocessed samples
        printf("Warning: Failed to load reference samples. Visualization will be limited.\n");
    }

    // Initialize drawing UI
    DrawingUI ui;
    if (!initUI(&ui, &model, numClasses, recognizeLetters)) {
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
    freeNaiveBayes(&model);

    return 0;
}