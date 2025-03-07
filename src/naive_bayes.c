//basic naive bayes implementation 

void initNaiveBayes(NaiveBayesModel *model, int numBins, double alpha) {
    model->numBins = numBins;
    model->binWidth = 256 / numBins;
    model->alpha = alpha;

    //set probs to zero to start
    memset(model->pixelProb, 0, sizeof(model->pixelProb));
    memset(model->classPrior, 0, sizeof(model->classPrior));
}
//map pixel intensity to bin
int getBin(int intensity, int binWidth) {
    return intensity /binWidth;
}

void trainNaiveBayes(NaiveBayesModel *model, MNISTDataset *dataset) {
    int counts[10][784][256] = {0};
    int classCounts[10] = {0};
    
    // Go through all training images
    for (uint32_t i = 0; i < dataset->numImages; i++) {
        uint8_t label = dataset->labels[i];
        classCounts[label]++;
        
        // Count pixel intensities
        for (uint32_t j = 0; j < dataset->imageSize; j++) {
            uint8_t pixel = dataset->images[i * dataset->imageSize + j];
            int bin = getBin(pixel, model->binWidth);
            counts[label][j][bin]++;
        }
    }
    
    // Calculate class priors
    for (int c = 0; c < 10; c++) {
        model->classPrior[c] = (double)classCounts[c] / dataset->numImages;
    }
    
    // Calculate pixel probabilities with Laplace smoothing
    for (int c = 0; c < 10; c++) {
        for (uint32_t j = 0; j < dataset->imageSize; j++) {
            for (int b = 0; b < model->numBins; b++) {
                // Apply Laplace smoothing
                model->pixelProb[c][j][b] = 
                    (counts[c][j][b] + model->alpha) / 
                    (classCounts[c] + model->alpha * model->numBins);
            }
        }
    }
}