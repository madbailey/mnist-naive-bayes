#ifndef UI_DRAWER_H
#define UI_DRAWER_H

#include <SDL2/SDL.h>
#include "naive_bayes.h"
#include "hog.h"

// Flags for different visualization modes
#define VIZ_MODE_NONE 0
#define VIZ_MODE_PROCESSED 1
#define VIZ_MODE_REFERENCE 2
#define VIZ_MODE_HOG 3

// Structure to hold UI components
typedef struct {
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *canvasTexture;
    uint8_t canvas[28*28];         // 28x28 pixel canvas for drawing
    uint8_t processedCanvas[28*28]; // Processed version for debugging
    int vizMode;                    // Visualization mode flag
    int showProcessed;              // Flag to show processed view (for backward compatibility)
    int drawing;                    // Flag to track if we're currently drawing
    NaiveBayesModel *model;        // Pointer to our trained model
    int numClasses;                // Number of classes (10 for digits, 26 for letters)
    int showingLetters;            // 0 for digits, 1 for letters
    double confidence[26];         // Confidence scores for each class
    int prediction;                // Current prediction
    double *lastFeatures;          // Store last extracted features for visualization
    int lastFeaturesCount;         // Number of features stored
} DrawingUI;

// Structure to hold HOG visualization data
typedef struct {
    double featureMap[28][28];     // Mapped importance of each pixel
    int hasData;                   // Flag indicating if visualization data is available
} HOGVisualization;

// Structure to hold reference samples
typedef struct {
    uint8_t samples[26][3][28*28];  // 3 samples for each of the 26 letters
    int numSamplesPerClass;
    int loaded;
} ReferenceSamples;

// External declarations for global variables (defined in ui_drawer.c)
extern HOGVisualization gHOGViz;
extern ReferenceSamples gReferenceSamples;

// Initialize the drawing UI
int initUI(DrawingUI *ui, NaiveBayesModel *model, int numClasses, int showLetters);

// Clean up resources
void cleanupUI(DrawingUI *ui);

// Process events (mouse, keyboard, etc.)
int processEvents(DrawingUI *ui);

// Draw the canvas and UI elements
void renderUI(DrawingUI *ui);

// Clear the canvas
void clearCanvas(DrawingUI *ui);

// Process the current drawing and make a prediction
void processPrediction(DrawingUI *ui);

// Preprocess the canvas for better recognition
void preprocessCanvas(uint8_t *canvas, uint8_t *processedCanvas);

// Load reference samples from training data
int loadReferenceSamples(const char* imageFile, const char* labelFile);

// Visualize HOG features
void visualizeHOGFeatures(DrawingUI *ui, double *features, uint8_t predictedClass);

// Render HOG visualization
void renderHOGVisualization(SDL_Renderer *renderer, int x, int y, int size);

// Render reference samples
void renderReferenceSamples(SDL_Renderer *renderer, int x, int y, int width, int height, int letterIndex);

// Change visualization mode
void cycleVisualizationMode(DrawingUI *ui);

#endif // UI_DRAWER_H