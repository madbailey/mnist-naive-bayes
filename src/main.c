int main() {
    MNISTDataset trainDataset, testDataset;
    NaiveBayesModel model;
    
    // Load training data
    printf("Loading training data...\n");
    if (!loadMNISTDataset("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", &trainDataset)) {
        return 1;
    }
    printf("Loaded %u training images\n", trainDataset.numImages);
    
    // Load test data
    printf("Loading test data...\n");
    if (!loadMNISTDataset("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", &testDataset)) {
        freeMNISTDataset(&trainDataset);
        return 1;
    }
    printf("Loaded %u test images\n", testDataset.numImages);
    
    // Initialize and train the model
    printf("Training Naive Bayes model...\n");
    initNaiveBayes(&model, 32, 1.0);  // 32 bins, alpha=1.0
    trainNaiveBayes(&model, &trainDataset);
    
    // Test the model
    printf("Testing the model...\n");
    int correct = 0;
    for (uint32_t i = 0; i < testDataset.numImages; i++) {
        uint8_t prediction = predictNaiveBayes(&model, 
                                             &testDataset.images[i * testDataset.imageSize],
                                             testDataset.imageSize);
        
        if (prediction == testDataset.labels[i]) {
            correct++;
        }
        
        // Print progress
        if ((i + 1) % 1000 == 0 || i + 1 == testDataset.numImages) {
            printf("Progress: %u/%u, Accuracy: %.2f%%\n", 
                   i + 1, testDataset.numImages, 
                   100.0 * correct / (i + 1));
        }
    }
    
    // Final accuracy
    double accuracy = 100.0 * correct / testDataset.numImages;
    printf("Final accuracy: %.2f%%\n", accuracy);
    
    // Free memory
    freeMNISTDataset(&trainDataset);
    freeMNISTDataset(&testDataset);
    
    return 0;
}