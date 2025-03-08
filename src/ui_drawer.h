#ifndef UI_DRAWER_H
#define UI_DRAWER_H

#include <SDL2/SDL.h>
#include "naive_bayes.h"
#include "hog.h"

// Structure to hold UI components
typedef struct {
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *canvasTexture;
    uint8_t canvas[28*28];     // 28x28 pixel canvas for drawing
    int drawing;               // Flag to track if we're currently drawing
    NaiveBayesModel *model;    // Pointer to our trained model
    int numClasses;            // Number of classes (10 for digits, 26 for letters)
    int showingLetters;        // 0 for digits, 1 for letters
    double confidence[26];     // Confidence scores for each class
    int prediction;            // Current prediction
} DrawingUI;

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

#endif // UI_DRAWER_H