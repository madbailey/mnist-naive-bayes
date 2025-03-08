#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "ui_drawer.h"
#include "hog.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define CANVAS_SIZE 280  // 10x scale for the 28x28 image
#define CANVAS_X 50
#define CANVAS_Y 50

// Colors
SDL_Color WHITE = {255, 255, 255, 255};
SDL_Color BLACK = {0, 0, 0, 255};
SDL_Color GRAY = {200, 200, 200, 255};
SDL_Color LIGHT_GRAY = {240, 240, 240, 255};
SDL_Color RED = {255, 0, 0, 255};
SDL_Color GREEN = {0, 200, 0, 255};
SDL_Color BLUE = {0, 0, 255, 255};

// Font for rendering text
TTF_Font *gFont = NULL;

// Convert numeric label to character
char getLabelChar(int label, int showingLetters) {
    if (showingLetters) {
        // For letters: 0=A, 1=B, ..., 25=Z
        return 'A' + label;
    } else {
        // For digits: 0-9
        return '0' + label;
    }
}

// Initialize the drawing UI
int initUI(DrawingUI *ui, NaiveBayesModel *model, int numClasses, int showLetters) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return 0;
    }
    
    // Initialize SDL_ttf
    if (TTF_Init() == -1) {
        printf("SDL_ttf could not initialize! TTF_Error: %s\n", TTF_GetError());
        return 0;
    }
    
    // Load font
    gFont = NULL;
    const char* fontPaths[] = {
        "FreeSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/TTF/FreeSans.ttf",
        "/Library/Fonts/Arial.ttf",                    
        "C:\\Windows\\Fonts\\arial.ttf"                 
    };

    for (int i = 0; i < sizeof(fontPaths) / sizeof(fontPaths[0]); i++) {
        gFont = TTF_OpenFont(fontPaths[i], 24);
        if (gFont != NULL) {
            printf("Successfully loaded font from: %s\n", fontPaths[i]);
            break;
        }
    }

    if (gFont == NULL) {
        printf("ERROR: Failed to load any font! Text rendering will not work properly.\n");
    }
    
    // Create window
    ui->window = SDL_CreateWindow("Letter Recognizer", 
                                 SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                 WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (ui->window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return 0;
    }
    
    // Create renderer
    ui->renderer = SDL_CreateRenderer(ui->window, -1, SDL_RENDERER_ACCELERATED);
    if (ui->renderer == NULL) {
        printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        return 0;
    }
    
    // Create canvas texture
    ui->canvasTexture = SDL_CreateTexture(ui->renderer, SDL_PIXELFORMAT_RGBA8888,
                                        SDL_TEXTUREACCESS_TARGET, 28, 28);
    if (ui->canvasTexture == NULL) {
        printf("Canvas texture could not be created! SDL_Error: %s\n", SDL_GetError());
        return 0;
    }
    
    // Initialize other UI properties
    memset(ui->canvas, 0, 28*28);
    ui->drawing = 0;
    ui->model = model;
    ui->numClasses = numClasses;
    ui->showingLetters = showLetters;
    ui->prediction = -1;  // No prediction yet
    memset(ui->confidence, 0, sizeof(ui->confidence));
    
    // Clear the canvas
    clearCanvas(ui);
    
    return 1;
}

// Clean up resources
void cleanupUI(DrawingUI *ui) {
    if (ui->canvasTexture != NULL) {
        SDL_DestroyTexture(ui->canvasTexture);
        ui->canvasTexture = NULL;
    }
    
    if (ui->renderer != NULL) {
        SDL_DestroyRenderer(ui->renderer);
        ui->renderer = NULL;
    }
    
    if (ui->window != NULL) {
        SDL_DestroyWindow(ui->window);
        ui->window = NULL;
    }
    
    if (gFont != NULL) {
        TTF_CloseFont(gFont);
        gFont = NULL;
    }
    
    TTF_Quit();
    SDL_Quit();
}

// Render text with the loaded font
void renderText(SDL_Renderer *renderer, int x, int y, const char *text, SDL_Color color) {
    if (gFont == NULL) {
        // Simple text rendering if font failed to load
        // (This is very basic and doesn't actually show text, just a placeholder)
        SDL_Rect textRect = {x, y, 100, 30};
        SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
        SDL_RenderFillRect(renderer, &textRect);
        return;
    }
    
    SDL_Surface *textSurface = TTF_RenderText_Solid(gFont, text, color);
    if (textSurface == NULL) {
        printf("Unable to render text surface! TTF_Error: %s\n", TTF_GetError());
        return;
    }
    
    SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
    if (textTexture == NULL) {
        printf("Unable to create texture from text! SDL_Error: %s\n", SDL_GetError());
        SDL_FreeSurface(textSurface);
        return;
    }
    
    SDL_Rect renderRect = {x, y, textSurface->w, textSurface->h};
    SDL_RenderCopy(renderer, textTexture, NULL, &renderRect);
    
    SDL_FreeSurface(textSurface);
    SDL_DestroyTexture(textTexture);
}

// Draw a button
int drawButton(SDL_Renderer *renderer, int x, int y, int w, int h, const char *text, SDL_Color color) {
    SDL_Rect buttonRect = {x, y, w, h};
    
    // Draw button background and border
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
    SDL_RenderFillRect(renderer, &buttonRect);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderDrawRect(renderer, &buttonRect);

    
    // Render the button text using the already loaded gFont
    if (gFont != NULL) {
        SDL_Surface *textSurface = TTF_RenderText_Solid(gFont, text, (SDL_Color){0, 0, 0, 255});
        if (textSurface != NULL) {
            SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
            if (textTexture != NULL) {
                int textW = textSurface->w;
                int textH = textSurface->h;
                SDL_Rect textRect = { x + (w - textW) / 2, y + (h - textH) / 2, textW, textH };
                SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
                SDL_DestroyTexture(textTexture);
            }
            SDL_FreeSurface(textSurface);
        }
    }

    return 1;
}

// Check if point is inside button
int isInsideButton(int x, int y, int btnX, int btnY, int btnW, int btnH) {
    return (x >= btnX && x <= btnX + btnW && y >= btnY && y <= btnY + btnH);
}

// Process mouse and keyboard events
int processEvents(DrawingUI *ui) {
    SDL_Event e;
    
    while (SDL_PollEvent(&e) != 0) {
        if (e.type == SDL_QUIT) {
            return 0;  // Exit
        }
        else if (e.type == SDL_MOUSEBUTTONDOWN) {
            int x, y;
            SDL_GetMouseState(&x, &y);
            
            // Check if click is inside canvas
            if (x >= CANVAS_X && x < CANVAS_X + CANVAS_SIZE &&
                y >= CANVAS_Y && y < CANVAS_Y + CANVAS_SIZE) {
                ui->drawing = 1;
                
                // Convert mouse coordinates to canvas pixel coordinates
                int canvasX = (x - CANVAS_X) * 28 / CANVAS_SIZE;
                int canvasY = (y - CANVAS_Y) * 28 / CANVAS_SIZE;
                
                // Draw a 3x3 "brush" centered on the pixel
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int px = canvasX + dx;
                        int py = canvasY + dy;
                        if (px >= 0 && px < 28 && py >= 0 && py < 28) {
                            // Set pixel to maximum intensity (255)
                            ui->canvas[py * 28 + px] = 255;
                        }
                    }
                }
            }
            
            // Check if click is on "Clear" button
            if (isInsideButton(x, y, 50, 350, 100, 40)) {
                clearCanvas(ui);
            }
            
            // Check if click is on "Predict" button
            if (isInsideButton(x, y, 170, 350, 100, 40)) {
                processPrediction(ui);
            }
        }
        else if (e.type == SDL_MOUSEBUTTONUP) {
            ui->drawing = 0;
        }
        else if (e.type == SDL_MOUSEMOTION && ui->drawing) {
            int x, y;
            SDL_GetMouseState(&x, &y);
            
            // Check if mouse is inside canvas
            if (x >= CANVAS_X && x < CANVAS_X + CANVAS_SIZE &&
                y >= CANVAS_Y && y < CANVAS_Y + CANVAS_SIZE) {
                
                // Convert mouse coordinates to canvas pixel coordinates
                int canvasX = (x - CANVAS_X) * 28 / CANVAS_SIZE;
                int canvasY = (y - CANVAS_Y) * 28 / CANVAS_SIZE;
                
                // Draw a 3x3 "brush" centered on the pixel
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int px = canvasX + dx;
                        int py = canvasY + dy;
                        if (px >= 0 && px < 28 && py >= 0 && py < 28) {
                            // Set pixel to maximum intensity (255)
                            ui->canvas[py * 28 + px] = 255;
                        }
                    }
                }
            }
        }
    }
    
    return 1;  // Continue running
}

// Draw the canvas and UI elements
void renderUI(DrawingUI *ui) {
    // Clear the renderer
    SDL_SetRenderDrawColor(ui->renderer, 240, 240, 240, 255);
    SDL_RenderClear(ui->renderer);
    
    // Draw canvas background
    SDL_Rect canvasRect = {CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE};
    SDL_SetRenderDrawColor(ui->renderer, 255, 255, 255, 255);
    SDL_RenderFillRect(ui->renderer, &canvasRect);
    
    // Draw canvas border
    SDL_SetRenderDrawColor(ui->renderer, 0, 0, 0, 255);
    SDL_RenderDrawRect(ui->renderer, &canvasRect);
    
    // Draw the pixels from our canvas data
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            if (ui->canvas[y * 28 + x] > 0) {
                // Scale up to display size
                SDL_Rect pixelRect = {
                    CANVAS_X + x * CANVAS_SIZE / 28,
                    CANVAS_Y + y * CANVAS_SIZE / 28,
                    CANVAS_SIZE / 28 + 1,  // +1 to avoid gaps
                    CANVAS_SIZE / 28 + 1
                };
                
                // Set color based on pixel intensity
                int intensity = ui->canvas[y * 28 + x];
                SDL_SetRenderDrawColor(ui->renderer, 0, 0, 0, intensity);
                SDL_RenderFillRect(ui->renderer, &pixelRect);
            }
        }
    }
    
    // Draw the "Clear" button
    drawButton(ui->renderer, 50, 350, 100, 40, "Clear", LIGHT_GRAY);
    
    // Draw the "Predict" button
    drawButton(ui->renderer, 170, 350, 100, 40, "Predict", LIGHT_GRAY);
    
    // Display instructions
    renderText(ui->renderer, 350, 60, "Draw a letter in the box", BLACK);
    renderText(ui->renderer, 350, 90, "and click 'Predict'", BLACK);
    
    // If we have a prediction, show it
    if (ui->prediction >= 0) {
        char predText[100];
        char label = getLabelChar(ui->prediction, ui->showingLetters);
        
        sprintf(predText, "Prediction: %c", label);
        renderText(ui->renderer, 350, 140, predText, BLUE);
        
        sprintf(predText, "Confidence: %.2f%%", ui->confidence[ui->prediction] * 100.0);
        renderText(ui->renderer, 350, 170, predText, BLUE);
        
        // Show top 5 predictions with confidences
        renderText(ui->renderer, 350, 210, "Top Predictions:", BLACK);
        
        // Find top 5 confidence scores
        int topIndices[5] = {-1, -1, -1, -1, -1};
        double topConfidences[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        
        for (int i = 0; i < ui->numClasses; i++) {
            // Find where this confidence ranks
            for (int j = 0; j < 5; j++) {
                if (ui->confidence[i] > topConfidences[j]) {
                    // Shift everything down
                    for (int k = 4; k > j; k--) {
                        topIndices[k] = topIndices[k-1];
                        topConfidences[k] = topConfidences[k-1];
                    }
                    // Insert new value
                    topIndices[j] = i;
                    topConfidences[j] = ui->confidence[i];
                    break;
                }
            }
        }
        
        // Display top 5
        for (int i = 0; i < 5; i++) {
            if (topIndices[i] >= 0) {
                char topText[100];
                char topLabel = getLabelChar(topIndices[i], ui->showingLetters);
                sprintf(topText, "%c: %.2f%%", topLabel, topConfidences[i] * 100.0);
                renderText(ui->renderer, 370, 240 + i*30, topText, BLACK);
            }
        }
    }
    
    // Update the screen
    SDL_RenderPresent(ui->renderer);
}

// Clear the canvas
void clearCanvas(DrawingUI *ui) {
    memset(ui->canvas, 0, 28*28);
    ui->prediction = -1;  // Clear prediction
    memset(ui->confidence, 0, sizeof(ui->confidence));
}

// Process the current drawing and make a prediction
void processPrediction(DrawingUI *ui) {
    // canvas data is already in the format we need (28x28 grayscale)
    // Now extract HOG features
    
    // Create a temporary dataset to extract HOG features
    MNISTDataset tempDataset;
    tempDataset.numImages = 1;
    tempDataset.imageSize = 28*28;
    tempDataset.rows = 28;
    tempDataset.cols = 28;
    tempDataset.images = ui->canvas;
    tempDataset.labels = NULL;  // Not needed for prediction
    
    // Create HOG features structure
    HOGFeatures hogFeatures;
    hogFeatures.numImages = 1;
    hogFeatures.numFeatures = ui->model->numFeatures;
    
    // Allocate memory for features
    hogFeatures.features = (double*)malloc(hogFeatures.numFeatures * sizeof(double));
    if (hogFeatures.features == NULL) {
        printf("Failed to allocate memory for HOG features\n");
        return;
    }
    
    // Extract HOG features
    int cellSize = 4;  // Same as used in training
    int numBins = 9;   // Same as used in training
    extractHOGFeatures(&tempDataset, &hogFeatures, cellSize, numBins);
    
    // Make prediction using our model
    double *features = hogFeatures.features;
    uint8_t prediction = predictNaiveBayes(ui->model, features);
    
    // Store prediction
    ui->prediction = prediction;
    
    // Now compute confidence scores for all classes
    // (this is a simplified approach - we use log probabilities directly)
    double totalProb = 0.0;
    
    // Get raw log probabilities for all classes
    double logProbs[26] = {0.0};
    double maxLogProb = -INFINITY;
    
    for (int c = 0; c < ui->numClasses; c++) {
        double logProb = log(ui->model->classPrior[c]);
        
        for (int f = 0; f < ui->model->numFeatures; f++) {
            int bin = (int)(features[f] / ui->model->binWidth);
            if (bin >= 0 && bin < ui->model->numBins) {
                logProb += log(ui->model->featureProb[c][f][bin]);
            }
        }
        
        logProbs[c] = logProb;
        if (logProb > maxLogProb) {
            maxLogProb = logProb;
        }
    }
    
    // Convert log probabilities to actual probabilities using softmax
    for (int c = 0; c < ui->numClasses; c++) {
        ui->confidence[c] = exp(logProbs[c] - maxLogProb);
        totalProb += ui->confidence[c];
    }
    
    // Normalize to get confidence scores
    for (int c = 0; c < ui->numClasses; c++) {
        ui->confidence[c] /= totalProb;
    }
    
    // Free HOG features
    free(hogFeatures.features);
}