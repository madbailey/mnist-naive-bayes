
// Function to load an MNIST dataset
int loadMNISTDataset(const char *imageFilename, const char *labelFilename, MNISTDataset *dataset) {
    FILE *imageFile, *labelFile;
    uint32_t imageMagic, labelMagic, numLabels;
    
    // Open the image file
    imageFile = fopen(imageFilename, "rb");
    if (imageFile == NULL) {
        perror("Error opening MNIST image file");
        return 0;
    }
    
    // Open the label file
    labelFile = fopen(labelFilename, "rb");
    if (labelFile == NULL) {
        perror("Error opening MNIST label file");
        fclose(imageFile);
        return 0;
    }
    
    // Read image header
    fread(&imageMagic, sizeof(imageMagic), 1, imageFile);
    fread(&dataset->numImages, sizeof(dataset->numImages), 1, imageFile);
    fread(&dataset->rows, sizeof(dataset->rows), 1, imageFile);
    fread(&dataset->cols, sizeof(dataset->cols), 1, imageFile);
    
    // Read label header
    fread(&labelMagic, sizeof(labelMagic), 1, labelFile);
    fread(&numLabels, sizeof(numLabels), 1, labelFile);
    
    // Convert headers from big-endian
    imageMagic = convert_endian(imageMagic);
    dataset->numImages = convert_endian(dataset->numImages);
    dataset->rows = convert_endian(dataset->rows);
    dataset->cols = convert_endian(dataset->cols);
    
    labelMagic = convert_endian(labelMagic);
    numLabels = convert_endian(numLabels);
    
    // Verify the magic numbers
    if (imageMagic != 2051 || labelMagic != 2049) {
        printf("Invalid MNIST file format\n");
        fclose(imageFile);
        fclose(labelFile);
        return 0;
    }
    
    // Check if the number of images matches the number of labels
    if (dataset->numImages != numLabels) {
        printf("Number of images doesn't match number of labels\n");
        fclose(imageFile);
        fclose(labelFile);
        return 0;
    }
    
    // Calculate image size
    dataset->imageSize = dataset->rows * dataset->cols;
    
    // Allocate memory for images and labels
    dataset->images = (uint8_t*)malloc(dataset->numImages * dataset->imageSize);
    dataset->labels = (uint8_t*)malloc(dataset->numImages);
    
    if (dataset->images == NULL || dataset->labels == NULL) {
        printf("Memory allocation failed\n");
        fclose(imageFile);
        fclose(labelFile);
        free(dataset->images);  // Safe to call free on NULL
        free(dataset->labels);
        return 0;
    }
    
    // Read all images
    if (fread(dataset->images, 1, dataset->numImages * dataset->imageSize, imageFile) != 
        dataset->numImages * dataset->imageSize) {
        printf("Failed to read all image data\n");
        fclose(imageFile);
        fclose(labelFile);
        free(dataset->images);
        free(dataset->labels);
        return 0;
    }
    
    // Read all labels
    if (fread(dataset->labels, 1, dataset->numImages, labelFile) != dataset->numImages) {
        printf("Failed to read all label data\n");
        fclose(imageFile);
        fclose(labelFile);
        free(dataset->images);
        free(dataset->labels);
        return 0;
    }
    
    // Close files
    fclose(imageFile);
    fclose(labelFile);
    
    return 1;  // Success
}

// Function to free the dataset
void freeMNISTDataset(MNISTDataset *dataset) {
    free(dataset->images);
    free(dataset->labels);
    dataset->images = NULL;
    dataset->labels = NULL;
}