#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mnist_loader.h"
#include "hog.h"
#include "naive_bayes.h"
#include "utils.h"

int main() {
    MNISTDataset trainDataset, testDataset;
    HOGFeatures trainHOG, testHOG;
    NaiveBayesModel model;
    
    int cellSize = 4;
    int numBins = 9;

    // Load training data
    printf("Loading training data...\n");
    if (!loadMNISTDataset("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", &trainDataset)) {
        printf("Failed to load training data. Check that files exist in the data/ directory.\n");
        return 1;
    }
    printf("Loaded %u training images\n", trainDataset.numImages);
    
    // Load test data
    printf("Loading test data...\n");
    if (!loadMNISTDataset("data/t10k-images.idx3-ubyte", "data/t10k-labels-idx1-ubyte", &testDataset)) {
        printf("Failed to load test data. Check that files exist in the data/ directory.\n");
        freeMNISTDataset(&trainDataset);
        return 1;
    }
    printf("Loaded %u test images\n", testDataset.numImages);

    // Initialize HOG feature structures
    trainHOG.numImages = trainDataset.numImages;
    trainHOG.numFeatures = (trainDataset.rows/cellSize) * (trainDataset.cols/cellSize) * numBins;
    
    testHOG.numImages = testDataset.numImages;
    testHOG.numFeatures = (testDataset.rows/cellSize) * (testDataset.cols/cellSize) * numBins;

    // Extract the HOG features
    printf("Extracting HOG features from training data...\n");
    extractHOGFeatures(&trainDataset, &trainHOG, cellSize, numBins);

    printf("Extracting HOG features from test data...\n");
    extractHOGFeatures(&testDataset, &testHOG, cellSize, numBins);

    // Initialize and train the model
    printf("Training model...\n");
    if (!initNaiveBayes(&model, 10, trainHOG.numFeatures, 32, 1.0)) {
        printf("Failed to initialize Naive Bayes model\n");
        return 1;
    }
    
    // Fix: Use trainHOG instead of trainDataset
    trainNaiveBayes(&model, &trainHOG);
    
    // Test the model
    printf("Testing the model...\n");
    int correct = 0;
    for (uint32_t i = 0; i < testHOG.numImages; i++) {
        double *features = &testHOG.features[i * testHOG.numFeatures];
        uint8_t prediction = predictNaiveBayes(&model, features);
        
        if (prediction == testHOG.labels[i]) {
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
    
    // Free memory
    freeMNISTDataset(&trainDataset);
    freeMNISTDataset(&testDataset);
    freeHOGFeatures(&trainHOG);
    freeHOGFeatures(&testHOG);
    freeNaiveBayes(&model);
    
    return 0;
}