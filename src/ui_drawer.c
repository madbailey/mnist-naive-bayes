#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "ui_drawer.h"
#include "hog.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define CANVAS_SIZE 280  // 10x scale for the 28x28 image
#define CANVAS_X 50
#define CANVAS_Y 50

// Prediction constants
#define PREDICTION_DELAY 500   // milliseconds to wait after drawing stops before predicting
#define CELL_SIZE 4           // MUST match the cell size used in training
#define NUM_BINS 9

// Colors
SDL_Color WHITE = {255, 255, 255, 255};
SDL_Color BLACK = {0, 0, 0, 255};
SDL_Color GRAY = {200, 200, 200, 255};
SDL_Color LIGHT_GRAY = {240, 240, 240, 255};
SDL_Color RED = {255, 0, 0, 255};
SDL_Color GREEN = {0, 200, 0, 255};
SDL_Color BLUE = {0, 0, 255, 255};

// Global variables - these should be in the .c file, not the header
ReferenceSamples gReferenceSamples = {
    .numSamplesPerClass = 3,
    .loaded = 0
};

// HOG visualization data
HOGVisualization gHOGViz = {
    .hasData = 0
};

// Font for rendering text
TTF_Font *gFont = NULL;

// Flag to indicate if we need to attempt prediction
int canvasDirty = 0;
Uint32 lastDrawTime = 0;

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

    for (size_t i = 0; i < sizeof(fontPaths) / sizeof(fontPaths[0]); i++) {
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
    memset(ui->processedCanvas, 0, 28*28);  // Initialize processedCanvas
    ui->vizMode = VIZ_MODE_PROCESSED;        // Initialize visualization mode
    ui->showProcessed = 0;                  // Initialize showProcessed flag
    ui->drawing = 0;
    ui->model = model;
    ui->numClasses = numClasses;
    ui->showingLetters = showLetters;
    ui->prediction = -1;  // No prediction yet
    ui->lastFeatures = NULL;  // Initialize lastFeatures
    ui->lastFeaturesCount = 0;  // Initialize lastFeaturesCount
    memset(ui->confidence, 0, sizeof(ui->confidence));

    // Initialize prediction flags
    canvasDirty = 0;
    lastDrawTime = 0;

    // Verify feature dimensions are correct
    int expectedFeatures = (28/CELL_SIZE) * (28/CELL_SIZE) * NUM_BINS;
    if (model->numFeatures != expectedFeatures) {
        printf("WARNING: Feature dimension mismatch! Model: %d, Expected: %d\n",
              model->numFeatures, expectedFeatures);
        printf("This may cause prediction errors or crashes.\n");
    }

    // Clear the canvas
    clearCanvas(ui);

    return 1;
}

// Clean up resources
void cleanupUI(DrawingUI *ui) {
    // Free any allocated feature memory
    if (ui->lastFeatures != NULL) {
        free(ui->lastFeatures);
        ui->lastFeatures = NULL;
    }

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

// Check if we should attempt prediction
int shouldPredict(void) {
    if (!canvasDirty)
        return 0;

    Uint32 currentTime = SDL_GetTicks();
    return (currentTime - lastDrawTime > PREDICTION_DELAY);
}

// Process mouse and keyboard events
int processEvents(DrawingUI *ui) {
    SDL_Event e;
    int wasDrawing = ui->drawing;

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

                // Reset prediction state when drawing starts
                if (!wasDrawing) {
                    ui->prediction = -1;  // Clear prediction
                    memset(ui->confidence, 0, sizeof(ui->confidence));
                    ui->showProcessed = 0; // Hide processed view when drawing new character
                }

                // Convert mouse coordinates to canvas pixel coordinates
                int canvasX = (x - CANVAS_X) * 28 / CANVAS_SIZE;
                int canvasY = (y - CANVAS_Y) * 28 / CANVAS_SIZE;

                // Draw a 3x3 "brush" centered on the pixel
                for (int dy = -0; dy <= 1; dy++) {
                    for (int dx = 0; dx <= 1; dx++) {
                        int px = canvasX + dx;
                        int py = canvasY + dy;
                        if (px >= 0 && px < 28 && py >= 0 && py < 28) {
                            // Set pixel to maximum intensity (255)
                            ui->canvas[py * 28 + px] = 255;
                        }
                    }
                }

                // Mark canvas as dirty and update last draw time
                canvasDirty = 1;
                lastDrawTime = SDL_GetTicks();
            }

            // Check if click is on "Clear" button
            if (isInsideButton(x, y, 50, 350, 100, 40)) {
                clearCanvas(ui);
            }

            // Check if click is on "Show/Hide Processed" button
            if (isInsideButton(x, y, 170, 350, 140, 40)) {
                ui->showProcessed = !ui->showProcessed;
            }
            // Check if click is on "Viz Mode" button
            if (isInsideButton(x, y, 330, 350, 150, 40)) {
                cycleVisualizationMode(ui);
                printf("Visualization mode changed to: %d\n", ui->vizMode);
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

                // Mark canvas as dirty and update last draw time
                canvasDirty = 1;
                lastDrawTime = SDL_GetTicks();
            }
        }
        else if (e.type == SDL_KEYDOWN) {
            // Press 'T' to toggle between showing the processed and original canvas
            if (e.key.keysym.sym == SDLK_t) {
                ui->showProcessed = !ui->showProcessed;
            }
        }
    }

    // Check if we should attempt prediction
    if (shouldPredict()) {
        processPrediction(ui);
        canvasDirty = 0;  // Canvas has been processed
    }

    return 1;  // Continue running
}

// Function to cycle through visualization modes
void cycleVisualizationMode(DrawingUI *ui) {
    ui->vizMode = (ui->vizMode + 1) % 4;  // Cycle through the 4 modes

    // Update button text based on new mode
    switch (ui->vizMode) {
        case VIZ_MODE_NONE:
            printf("Visualization mode: None\n");
            break;
        case VIZ_MODE_PROCESSED:
            printf("Visualization mode: Processed Image\n");
            break;
        case VIZ_MODE_REFERENCE:
            printf("Visualization mode: Reference Samples\n");
            break;
        case VIZ_MODE_HOG:
            printf("Visualization mode: HOG Features\n");
            break;
    }
}

// Draw the canvas and UI elements
void renderUI(DrawingUI *ui) {
    // Clear the renderer
    SDL_SetRenderDrawColor(ui->renderer, 240, 240, 240, 255);
    SDL_RenderClear(ui->renderer);

    // Draw canvas background (original canvas)
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

    // If we have a processed canvas, show it
    if (ui->showProcessed) {
        // Draw processed canvas background
        SDL_Rect processedRect = {CANVAS_X, CANVAS_Y + CANVAS_SIZE + 20, CANVAS_SIZE, CANVAS_SIZE};
        SDL_SetRenderDrawColor(ui->renderer, 255, 255, 255, 255);
        SDL_RenderFillRect(ui->renderer, &processedRect);

        // Draw processed canvas border
        SDL_SetRenderDrawColor(ui->renderer, 0, 0, 0, 255);
        SDL_RenderDrawRect(ui->renderer, &processedRect);

        // Draw processed pixels
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                if (ui->processedCanvas[y * 28 + x] > 0) {
                    // Scale up to display size
                    SDL_Rect pixelRect = {
                        CANVAS_X + x * CANVAS_SIZE / 28,
                        (CANVAS_Y + CANVAS_SIZE + 20) + y * CANVAS_SIZE / 28,
                        CANVAS_SIZE / 28 + 1,  // +1 to avoid gaps
                        CANVAS_SIZE / 28 + 1
                    };

                    // Set color based on pixel intensity
                    int intensity = ui->processedCanvas[y * 28 + x];
                    SDL_SetRenderDrawColor(ui->renderer, 0, 0, 0, intensity);
                    SDL_RenderFillRect(ui->renderer, &pixelRect);
                }
            }
        }

        // Label the processed canvas
        renderText(ui->renderer, CANVAS_X, CANVAS_Y + CANVAS_SIZE + 5, "Preprocessed", BLACK);
    }

    // Draw the "Clear" button
    drawButton(ui->renderer, 50, 350, 100, 40, "Clear", LIGHT_GRAY);

    // Draw toggle button for processed view
    drawButton(ui->renderer, 170, 350, 140, 40,
               ui->showProcessed ? "Hide Processed" : "Show Processed",
               ui->showProcessed ? GREEN : LIGHT_GRAY);

    // Display instructions
    renderText(ui->renderer, 350, 60, "Draw a letter in the box", BLACK);
    if (canvasDirty && !ui->drawing) {
        // Show that we're waiting to predict
        Uint32 currentTime = SDL_GetTicks();
        Uint32 timeLeft = (lastDrawTime + PREDICTION_DELAY) - currentTime;
        char waitText[64];
        sprintf(waitText, "Predicting in %.1f sec...", timeLeft / 1000.0);
        renderText(ui->renderer, 350, 90, waitText, BLUE);
    } else if (!canvasDirty) {
        renderText(ui->renderer, 350, 90, "Predictions are automatic", BLACK);
    }

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
        double topConfidences[5] = {0.0, 0.0, 0.0, 0.0};

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
    // Draw visualization mode selection button
    drawButton(ui->renderer, 330, 350, 150, 40, 
        "Viz Mode", LIGHT_GRAY);

    // Display current visualization mode
    char vizModeText[64];
    switch (ui->vizMode) {
    case VIZ_MODE_NONE:
    sprintf(vizModeText, "Mode: None");
    break;
    case VIZ_MODE_PROCESSED:
    sprintf(vizModeText, "Mode: Processed");
    break;
    case VIZ_MODE_REFERENCE:
    sprintf(vizModeText, "Mode: Reference");
    break;
    case VIZ_MODE_HOG:
    sprintf(vizModeText, "Mode: HOG Features");
    break;
    }
    renderText(ui->renderer, 330, 400, vizModeText, BLACK);

    // If we have a prediction, show the appropriate visualization based on mode
    if (ui->prediction >= 0) {
    switch (ui->vizMode) {
    case VIZ_MODE_NONE:
        // No visualization
        break;
        
    case VIZ_MODE_PROCESSED:
        // Processed view is already handled by the showProcessed flag
        ui->showProcessed = 1;
        break;
        
    case VIZ_MODE_REFERENCE:
        // Show reference samples for the predicted letter
        if (gReferenceSamples.loaded) {
            renderReferenceSamples(ui->renderer, 
                                    350, 450, 
                                    300, 100, 
                                    ui->prediction);
        } else {
            renderText(ui->renderer, 350, 450, 
                        "Reference samples not available", RED);
        }
        break;
        
    case VIZ_MODE_HOG:
        // Calculate and show HOG feature visualization
        if (ui->lastFeatures == NULL) {
            // Allocate space for features if not already done
            ui->lastFeatures = (double*)malloc(ui->model->numFeatures * sizeof(double));
            if (ui->lastFeatures == NULL) {
                renderText(ui->renderer, 350, 450, 
                            "Failed to allocate memory for feature viz", RED);
                break;
            }
            
            // We'll store features during processPrediction()
            ui->lastFeaturesCount = ui->model->numFeatures;
        }
        
        // Visualize the HOG features
        if (ui->lastFeatures != NULL && gHOGViz.hasData) {
            renderHOGVisualization(ui->renderer, 350, 450, 200);
        } else {
            renderText(ui->renderer, 350, 450, 
                        "HOG visualization not available", RED);
            renderText(ui->renderer, 350, 480, 
                        "Draw a new letter to generate", BLACK);
        }
        break;
    }
    }

    // Update the screen
    SDL_RenderPresent(ui->renderer);
}

// Clear the canvas
void clearCanvas(DrawingUI *ui) {
    memset(ui->canvas, 0, 28*28);
    memset(ui->processedCanvas, 0, 28*28);  // Also clear the processed canvas
    ui->showProcessed = 0;                  // Hide the processed view
    ui->prediction = -1;                    // Clear prediction
    memset(ui->confidence, 0, sizeof(ui->confidence));
    canvasDirty = 0;                        // Canvas is clean
}

void preprocessCanvas(uint8_t *canvas, uint8_t *processedCanvas) {
    // Step 1: Find the bounding box of the drawn character
    int minX = 28, minY = 28, maxX = 0, maxY = 0;
    int hasContent = 0;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            if (canvas[y * 28 + x] > 50) {
                hasContent = 1;
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
            }
        }
    }

    // If no content, return empty canvas
    if (!hasContent) {
        memset(processedCanvas, 0, 28 * 28);
        return;
    }

    // Step 2: Initialize processed canvas to zeros
    memset(processedCanvas, 0, 28 * 28);

    // Step 3: Calculate dimensions and center offset
    int width = maxX - minX + 1;
    int height = maxY - minY + 1;
    int size = (width > height) ? width : height;

    // Add padding (20% of size)
    int padding = size / 5;
    size += padding * 2;

    // Ensure size doesn't exceed canvas dimensions
    if (size > 28) size = 28;

    // Calculate centering offsets
    int offsetX = (28 - size) / 2;
    int offsetY = (28 - size) / 2;

    // Step 4: Scale and center the content
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            // Map the destination (x,y) back to source coordinates
            int srcX = minX + (x * width) / size;
            int srcY = minY + (y * height) / size;

            // Ensure source coordinates are in bounds
            if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {
                // Copy pixel to the centered position in the processed canvas
                processedCanvas[(offsetY + y) * 28 + (offsetX + x)] = canvas[srcY * 28 + srcX];
            }
        }
    }

    // Step 5: Apply thresholding and normalization
    for (int i = 0; i < 28 * 28; i++) {
        // Binary thresholding
        processedCanvas[i] = (processedCanvas[i] > 30) ? 255 : 0;
    }
}
// Replace the entire visualizeHOGFeatures() function with this improved version
void visualizeHOGFeatures(DrawingUI *ui, double *features, uint8_t predictedClass) {
    // Clear the visualization
    memset(&gHOGViz.featureMap, 0, sizeof(gHOGViz.featureMap));
    gHOGViz.hasData = 0;  // Set to 0 initially, will set to 1 when successful
    
    // Early return if invalid inputs
    if (features == NULL || ui->model == NULL) {
        printf("Invalid inputs for HOG visualization\n");
        return;
    }
    
    // Get parameters
    int cellSize = CELL_SIZE;  // IMPORTANT: Must match training
    int numBins = NUM_BINS;
    int cellsX = 28 / cellSize;
    int cellsY = 28 / cellSize;
    
    // Create an array to store importance of each feature
    double *featureImportance = (double*)malloc(ui->model->numFeatures * sizeof(double));
    if (featureImportance == NULL) {
        printf("Failed to allocate memory for feature importance\n");
        return;
    }
    
    // Calculate feature importance for the predicted class
    for (int f = 0; f < ui->model->numFeatures; f++) {
        double featureVal = features[f];
        
        // Ensure feature value is in valid range
        featureVal = (featureVal < 0) ? 0 : (featureVal > 1.0 ? 1.0 : featureVal);
        
        // Determine which bin the orientation falls into
        int bin = (int)(featureVal / ui->model->binWidth);
        bin = (bin < 0) ? 0 : (bin >= ui->model->numBins ? ui->model->numBins - 1 : bin);
        
        // Calculate importance based on likelihood ratio
        double importance = 0;
        
        // Compare this feature's probability for the predicted class vs. average of other classes
        double probForClass = ui->model->featureProb[predictedClass][f][bin];
        double avgProbOtherClasses = 0;
        int numOtherClasses = 0;
        
        for (int c = 0; c < ui->model->numClasses; c++) {
            if (c != predictedClass) {
                avgProbOtherClasses += ui->model->featureProb[c][f][bin];
                numOtherClasses++;
            }
        }
        
        // Calculate average probability for other classes
        if (numOtherClasses > 0) {
            avgProbOtherClasses /= numOtherClasses;
        }
        
        // Calculate importance as a ratio (avoid division by zero)
        if (avgProbOtherClasses > 1e-10) {
            importance = probForClass / avgProbOtherClasses;
        } else {
            importance = probForClass > 1e-10 ? 10.0 : 1.0;  // Arbitrary high value if unique to this class
        }
        
        // Take log to handle wide range of values
        importance = log(importance + 1.0);  // +1 to avoid negative values for ratios < 1
        
        // Store importance
        featureImportance[f] = importance;
    }
    
    // Map feature importance back to image pixels
    for (int f = 0; f < ui->model->numFeatures; f++) {
        // Calculate which cell this feature belongs to
        int binIndex = f % numBins;
        int cellIndex = f / numBins;
        int cellY = cellIndex / cellsX;
        int cellX = cellIndex % cellsX;
        
        // Skip if cell coordinates are invalid
        if (cellY >= cellsY || cellX >= cellsX) {
            continue;
        }
        
        // For each pixel in this cell, add the feature importance
        for (int y = 0; y < cellSize; y++) {
            for (int x = 0; x < cellSize; x++) {
                int pixelY = cellY * cellSize + y;
                int pixelX = cellX * cellSize + x;
                
                if (pixelY < 28 && pixelX < 28) {
                    // Scale by bin index to visualize orientation
                    // This will make different orientations appear with different intensities
                    double scaledImportance = featureImportance[f] * (1.0 + 0.2 * binIndex);
                    gHOGViz.featureMap[pixelY][pixelX] += scaledImportance;
                }
            }
        }
    }
    
    // Normalize the feature map to [0, 1] range
    double minVal = 0;
    double maxVal = 0;
    int hasNonZeroValues = 0;
    
    // Find min and max values
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            if (!hasNonZeroValues || gHOGViz.featureMap[y][x] != 0) {
                if (!hasNonZeroValues) {
                    minVal = maxVal = gHOGViz.featureMap[y][x];
                    hasNonZeroValues = 1;
                } else {
                    if (gHOGViz.featureMap[y][x] < minVal) minVal = gHOGViz.featureMap[y][x];
                    if (gHOGViz.featureMap[y][x] > maxVal) maxVal = gHOGViz.featureMap[y][x];
                }
            }
        }
    }
    
    // Normalize if we have a valid range
    if (hasNonZeroValues && maxVal > minVal) {
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                // Normalize to [0, 1]
                gHOGViz.featureMap[y][x] = (gHOGViz.featureMap[y][x] - minVal) / (maxVal - minVal);
            }
        }
        gHOGViz.hasData = 1;  // Mark as successful
    }
    
    // Free temporary memory
    free(featureImportance);
    
    printf("HOG feature visualization created (min=%f, max=%f)\n", minVal, maxVal);
}
// Load reference samples from the training dataset
int loadReferenceSamples(const char* imageFile, const char* labelFile) {
    MNISTDataset refDataset;
    int isEMNIST = (strstr(imageFile, "emnist") != NULL);

    // Load the dataset with the appropriate function
    if (isEMNIST) {
        printf("Loading EMNIST reference samples...\n");
        if (!loadEMNISTDataset(imageFile, labelFile, &refDataset)) {
            printf("Failed to load reference samples\n");
            return 0;
        }
    } else {
        printf("Loading MNIST reference samples...\n");
        if (!loadMNISTDataset(imageFile, labelFile, &refDataset)) {
            printf("Failed to load reference samples\n");
            return 0;
        }
    }

    // Adjust labels to be 0-based for EMNIST
    if (isEMNIST) {
        printf("Adjusting reference sample labels to be 0-based...\n");
        for (uint32_t i = 0; i < refDataset.numImages; i++) {
            if (refDataset.labels[i] > 0) {
                refDataset.labels[i] -= 1;  // Make 1-26 into 0-25
            }
        }
    }

    // Initialize the reference samples
    memset(gReferenceSamples.samples, 0, sizeof(gReferenceSamples.samples));
    gReferenceSamples.numSamplesPerClass = 3;
    gReferenceSamples.loaded = 1;

    // Keep track of how many samples we've found for each class
    int sampleCounts[26] = {0};
    int maxClassesToLoad = isEMNIST ? 26 : 10;

    // Go through the dataset and pick representative samples
    for (uint32_t i = 0; i < refDataset.numImages && i < 5000; i++) {  // Limit to first 5000 images for speed
        uint8_t label = refDataset.labels[i];

        // Skip if label is out of range for our application
        if (label >= maxClassesToLoad) continue;

        // If we haven't filled this class yet
        if (sampleCounts[label] < gReferenceSamples.numSamplesPerClass) {
            // Copy this image as a reference sample
            memcpy(gReferenceSamples.samples[label][sampleCounts[label]],
                   &refDataset.images[i * refDataset.imageSize],
                   refDataset.imageSize);

            sampleCounts[label]++;
        }

        // Break if we've filled all classes
        int allFilled = 1;
        for (int c = 0; c < maxClassesToLoad; c++) {
            if (sampleCounts[c] < gReferenceSamples.numSamplesPerClass) {
                allFilled = 0;
                break;
            }
        }
        if (allFilled) break;
    }

    // Print how many samples we found
    printf("Reference samples loaded:\n");
    for (int c = 0; c < maxClassesToLoad; c++) {
        if (isEMNIST) {
            printf("%c: %d, ", 'A' + c, sampleCounts[c]);
        } else {
            printf("%d: %d, ", c, sampleCounts[c]);
        }
        if ((c+1) % 6 == 0) printf("\n");
    }
    printf("\n");

    // Free the dataset
    freeMNISTDataset(&refDataset);

    return 1;
}
// Replace the renderHOGVisualization() function with this version
void renderHOGVisualization(SDL_Renderer *renderer, int x, int y, int size) {
    if (!gHOGViz.hasData) {
        // Draw placeholder if we don't have data
        SDL_Rect rect = {x, y, size, size};
        SDL_SetRenderDrawColor(renderer, 200, 200, 200, 255);
        SDL_RenderFillRect(renderer, &rect);
        renderText(renderer, x + 20, y + size/2 - 10, "No HOG data", BLACK);
        return;
    }
    
    // Draw title
    renderText(renderer, x, y - 30, "HOG Feature Importance", BLACK);
    renderText(renderer, x, y - 10, "Red = Strong Feature for This Class", BLACK);
    
    // Draw background
    SDL_Rect bgRect = {x, y, size, size};
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderFillRect(renderer, &bgRect);
    
    // Draw the HOG heatmap
    for (int py = 0; py < 28; py++) {
        for (int px = 0; px < 28; px++) {
            double val = gHOGViz.featureMap[py][px];
            
            // Only draw significant values
            if (val < 0.05) continue;
            
            // Scale pixel coordinates to display size
            int dispX = x + px * size / 28;
            int dispY = y + py * size / 28;
            int pixSize = size / 28 + 1;  // +1 to avoid gaps
            
            // Create a color based on importance
            // Red = high importance, Blue = low importance
            uint8_t r = (uint8_t)(val * 255);
            uint8_t g = 0;
            uint8_t b = (uint8_t)((1.0 - val) * 255);
            uint8_t a = (uint8_t)(val * 200 + 55);  // More important = more opaque
            
            // Draw the pixel
            SDL_Rect pixRect = {dispX, dispY, pixSize, pixSize};
            SDL_SetRenderDrawColor(renderer, r, g, b, a);
            SDL_RenderFillRect(renderer, &pixRect);
        }
    }
    
    // Draw original image overlay (faint)
    for (int py = 0; py < 28; py++) {
        for (int px = 0; px < 28; px++) {
            if (gHOGViz.hasData && py < 28 && px < 28) {
                int dispX = x + px * size / 28;
                int dispY = y + py * size / 28;
                int pixSize = size / 28 + 1;
                
                // Only draw if the pixel is set in the original image
                if (gHOGViz.hasData && 
                    gHOGViz.featureMap[py][px] < 0.05) { // Low importance original pixels
                    
                    SDL_Rect pixRect = {dispX, dispY, pixSize, pixSize};
                    SDL_SetRenderDrawColor(renderer, 100, 100, 100, 40);  // Very faint gray
                    SDL_RenderFillRect(renderer, &pixRect);
                }
            }
        }
    }
    
    // Draw border around the visualization
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderDrawRect(renderer, &bgRect);
}
// Display reference samples for comparison
void renderReferenceSamples(SDL_Renderer *renderer, int x, int y, int width, int height, int letterIndex) {
    if (!gReferenceSamples.loaded || letterIndex < 0 || letterIndex >= 26) {
        return;
    }
    
    // Draw background and title
    SDL_Rect bgRect = {x, y, width, height};
    SDL_SetRenderDrawColor(renderer, 240, 240, 240, 255);
    SDL_RenderFillRect(renderer, &bgRect);
    
    char title[64];
    sprintf(title, "Reference '%c' Samples", 'A' + letterIndex);
    renderText(renderer, x, y - 20, title, (SDL_Color){0, 0, 0, 255});
    
    // Calculate sample size and spacing
    int sampleSize = width / gReferenceSamples.numSamplesPerClass;
    int spacing = 10;
    
    // Draw each sample
    for (int i = 0; i < gReferenceSamples.numSamplesPerClass; i++) {
        int sampleX = x + i * (sampleSize + spacing);
        
        // Draw sample background
        SDL_Rect sampleRect = {sampleX, y, sampleSize, sampleSize};
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderFillRect(renderer, &sampleRect);
        
        // Draw sample border
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderDrawRect(renderer, &sampleRect);
        
        // Draw the sample pixels
        for (int sy = 0; sy < 28; sy++) {
            for (int sx = 0; sx < 28; sx++) {
                if (gReferenceSamples.samples[letterIndex][i][sy * 28 + sx] > 50) {
                    SDL_Rect pixelRect = {
                        sampleX + sx * sampleSize / 28,
                        y + sy * sampleSize / 28,
                        sampleSize / 28 + 1,
                        sampleSize / 28 + 1
                    };
                    
                    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
                    SDL_RenderFillRect(renderer, &pixelRect);
                }
            }
        }
    }
}
// Process the current drawing and make a prediction
void processPrediction(DrawingUI *ui) {
    // Preprocess the canvas
    uint8_t processedCanvas[28*28];
    preprocessCanvas(ui->canvas, processedCanvas);
    
    // Get the correct cellSize - MUST match what was used in training
    int cellSize = CELL_SIZE;
    int numBins = NUM_BINS;
    
    // IMPORTANT: Calculate numFeatures the same way it was calculated during training
    int numFeatures = (28/cellSize) * (28/cellSize) * numBins;
    
    // Verify feature dimensions match the model
    if (numFeatures != ui->model->numFeatures) {
        printf("ERROR: Feature dimension mismatch! Expected: %d, Got: %d\n", 
               ui->model->numFeatures, numFeatures);
        printf("This is likely due to a cell size mismatch between training and prediction.\n");
        return;
    }
    
    // Create a temporary dataset to extract HOG features
    MNISTDataset tempDataset;
    tempDataset.numImages = 1;
    tempDataset.imageSize = 28*28;
    tempDataset.rows = 28;
    tempDataset.cols = 28;
    tempDataset.images = processedCanvas;  // Use the processed image
    tempDataset.labels = NULL;  // Not needed for prediction
    
    // Create HOG features structure and initialize all fields
    HOGFeatures hogFeatures;
    hogFeatures.numImages = 1;
    hogFeatures.numFeatures = numFeatures;  // Use the calculated value
    hogFeatures.labels = NULL; 
    
    // Allocate memory for features
    hogFeatures.features = (double*)malloc(hogFeatures.numFeatures * sizeof(double));
    if (hogFeatures.features == NULL) {
        printf("Failed to allocate memory for HOG features\n");
        return;
    }
    
    // Initialize features memory to zeros
    memset(hogFeatures.features, 0, hogFeatures.numFeatures * sizeof(double));
    
    // Extract HOG features using the correct cell size
    extractHOGFeatures(&tempDataset, &hogFeatures, cellSize, numBins);
    
    // Store all log probabilities
    double *logProbs = (double*)malloc(ui->numClasses * sizeof(double));
    if (logProbs == NULL) {
        printf("Failed to allocate memory for log probabilities\n");
        free(hogFeatures.features);
        return;
    }
    
    // Calculate log probabilities directly
    double maxLogProb = -INFINITY;
    int bestClass = 0;
    
    for (int c = 0; c < ui->numClasses; c++) {
        // Start with class prior probability
        double logProb = log(ui->model->classPrior[c]);
        
        // Add log probability for each feature
        for (int f = 0; f < ui->model->numFeatures; f++) {
            // Ensure feature value is in valid range
            double featureVal = hogFeatures.features[f];
            featureVal = (featureVal < 0) ? 0 : (featureVal > 1.0 ? 1.0 : featureVal);
            
            // Calculate bin index
            int bin = (int)(featureVal / ui->model->binWidth);
            bin = (bin < 0) ? 0 : (bin >= ui->model->numBins ? ui->model->numBins - 1 : bin);
            
            // Get probability for this feature, with safety check
            double prob = ui->model->featureProb[c][f][bin];
            prob = (prob < 1e-10) ? 1e-10 : prob;  // Prevent log(0)
            
            logProb += log(prob);
        }
        
        // Store this class's log probability
        logProbs[c] = logProb;
        
        // Keep track of best class
        if (logProb > maxLogProb) {
            maxLogProb = logProb;
            bestClass = c;
        }
    }
    
    // Set prediction
    ui->prediction = bestClass;
    
    // Apply a temperature parameter to soften the confidence scores
    double temperature = 2.5;  // Higher values = softer distribution
    double totalProb = 0.0;
    
    // Convert log probabilities to actual probabilities using softmax with temperature
    for (int c = 0; c < ui->numClasses; c++) {
        ui->confidence[c] = exp((logProbs[c] - maxLogProb) / temperature);
        totalProb += ui->confidence[c];
    }
    
    // Normalize to get confidence scores
    if (totalProb > 0) {
        for (int c = 0; c < ui->numClasses; c++) {
            ui->confidence[c] /= totalProb;
        }
    }
    
    // Copy processed canvas to a separate place to display for debugging
    memcpy(ui->processedCanvas, processedCanvas, 28*28);
    ui->showProcessed = 1;
    
    // Print the prediction for debugging
    printf("Predicted: %c with confidence %.2f%%\n", 
          ui->showingLetters ? 'A' + ui->prediction : '0' + ui->prediction,
          ui->confidence[ui->prediction] * 100.0);
    
    // Store the extracted features for visualization if in HOG mode
    if (ui->vizMode == VIZ_MODE_HOG) {
        // Allocate/reallocate memory for features if needed
        if (ui->lastFeatures == NULL || ui->lastFeaturesCount != hogFeatures.numFeatures) {
            // Free old memory if it exists
            if (ui->lastFeatures != NULL) {
                free(ui->lastFeatures);
            }
            
            // Allocate new memory
            ui->lastFeatures = (double*)malloc(hogFeatures.numFeatures * sizeof(double));
            if (ui->lastFeatures != NULL) {
                ui->lastFeaturesCount = hogFeatures.numFeatures;
            } else {
                ui->lastFeaturesCount = 0;
                printf("Failed to allocate memory for feature storage\n");
            }
        }
        
        // Copy features if memory allocation succeeded
        if (ui->lastFeatures != NULL) {
            memcpy(ui->lastFeatures, hogFeatures.features, 
                hogFeatures.numFeatures * sizeof(double));
                
            // Generate the HOG visualization
            visualizeHOGFeatures(ui, hogFeatures.features, bestClass);
        }
    }


    
    // Free allocated memory in reverse order of allocation
    free(logProbs);
    free(hogFeatures.features);
    // Add this right at the end of processPrediction() before returning
    // This makes sure we keep the processed view visible for better user feedback
    ui->showProcessed = 1;
}