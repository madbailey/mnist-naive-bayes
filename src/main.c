#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mnist_loader.h"
#include "hog.h"
#include "naive_bayes.h"
#include "utils.h"

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
    HOGFeatures trainHOG, testHOG;
    NaiveBayesModel model;
    
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

    // Initialize HOG feature structures
    trainHOG.numImages = trainDataset.numImages;
    trainHOG.numFeatures = (trainDataset.rows/cellSize) * (trainDataset.cols/cellSize) * numBins;
    
    testHOG.numImages = testDataset.numImages;
    testHOG.numFeatures = (testDataset.rows/cellSize) * (testDataset.cols/cellSize) * numBins;

    // Extract HOG features
    printf("Extracting HOG features from training letters...\n");
    extractHOGFeatures(&trainDataset, &trainHOG, cellSize, numBins);

    printf("Extracting HOG features from test letters...\n");
    extractHOGFeatures(&testDataset, &testHOG, cellSize, numBins);

    // Initialize and train the model
    printf("Training letter recognition model...\n");
    if (!initNaiveBayes(&model, numClasses, trainHOG.numFeatures, 32, 1.0)) {
        printf("Failed to initialize Naive Bayes model\n");
        return 1;
    }
    
    trainNaiveBayes(&model, &trainHOG);
    
    // Test the model
    printf("Testing the letter recognition model...\n");
    int correct = 0;
    int confusionMatrix[26][26] = {0}; // Track misclassifications
    
    for (uint32_t i = 0; i < testHOG.numImages; i++) {
        double *features = &testHOG.features[i * testHOG.numFeatures];
        uint8_t prediction = predictNaiveBayes(&model, features);
        uint8_t actual = testHOG.labels[i];
        
        // Update confusion matrix (now with 0-based labels)
        if (actual < 26 && prediction < 26) {
            confusionMatrix[actual][prediction]++;
        }
        
        if (prediction == actual) {
            correct++;
        }
        
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
    
    // Free memory
    freeMNISTDataset(&trainDataset);
    freeMNISTDataset(&testDataset);
    freeHOGFeatures(&trainHOG);
    freeHOGFeatures(&testHOG);
    freeNaiveBayes(&model);
    
    return 0;
}