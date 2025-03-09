#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "raylib.h"
#include "ui_drawer.h"
#include "hog.h"
#include "normalization.h"

#define WINDOW_WIDTH 1200
#define WINDOW_HEIGHT 700
#define CANVAS_SIZE 350  // Scale for the 28x28 image
#define CANVAS_X 50
#define CANVAS_Y 50
#define PANEL_DIVIDER 450  // X position of divider between drawing panel and results panel

// Prediction constants
#define PREDICTION_DELAY 500   // milliseconds to wait after drawing stops before predicting
#define CELL_SIZE 4           // MUST match the cell size used in training
#define NUM_BINS 9

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
Font gFont;

// Flag to indicate if we need to attempt prediction
int canvasDirty = 0;
double lastDrawTime = 0;

// Canvas texture for drawing
RenderTexture2D canvasTexture;
RenderTexture2D processedCanvasTexture;



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
int initUI(DrawingUI *ui, NaiveBayesModel *model, SpecializedClassifierManager *manager, int numClasses, int showLetters) {
    // Initialize raylib window
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, showLetters ? "Glyph" : "Digit Recognizer");

    // Try to load font
    gFont = GetFontDefault();  // First use default font

    // Try to load a nicer font if available
    const char* fontPaths[] = {
        "FreeSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/TTF/FreeSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf"
    };

    for (size_t i = 0; i < sizeof(fontPaths) / sizeof(fontPaths[0]); i++) {
        if (FileExists(fontPaths[i])) {
            gFont = LoadFont(fontPaths[i]);
            printf("Successfully loaded font from: %s\n", fontPaths[i]);
            break;
        }
    }

    // Create render textures for the canvas and processed canvas
    canvasTexture = LoadRenderTexture(28, 28);
    processedCanvasTexture = LoadRenderTexture(28, 28);

    // Clear textures to white
    BeginTextureMode(canvasTexture);
    ClearBackground(RAYWHITE);
    EndTextureMode();

    BeginTextureMode(processedCanvasTexture);
    ClearBackground(RAYWHITE);
    EndTextureMode();

    // Initialize other UI properties
    memset(ui->canvas, 0, 28*28);
    memset(ui->processedCanvas, 0, 28*28);  // Initialize processedCanvas
    ui->vizMode = VIZ_MODE_PROCESSED;        // Initialize visualization mode
    ui->showProcessed = 0;                  // Initialize showProcessed flag
    ui->drawing = 0;
    ui->model = model;
    ui->specializedManager = manager;
    ui->numClasses = numClasses;
    ui->showingLetters = showLetters;
    ui->prediction = -1;  // No prediction yet
    ui->lastFeatures = NULL;  // Initialize lastFeatures
    ui->lastFeaturesCount = 0;  // Initialize lastFeaturesCount
    ui->selectedFeatureIndices = NULL;  // Initialize selected feature indices
    ui->numSelectedFeatures = 0;  // Initialize number of selected features
    memset(ui->confidence, 0, sizeof(ui->confidence));

    // Try to load selected feature indices if available
    FILE *featureIdxFile = fopen("selected_features.dat", "rb");
    if (featureIdxFile) {
        // Read number of selected features
        if (fread(&ui->numSelectedFeatures, sizeof(uint32_t), 1, featureIdxFile) == 1) {
            // Allocate memory for selected feature indices
            ui->selectedFeatureIndices = (uint32_t *)malloc(ui->numSelectedFeatures * sizeof(uint32_t));
            if (ui->selectedFeatureIndices) {
                // Read selected feature indices
                if (fread(ui->selectedFeatureIndices, sizeof(uint32_t), ui->numSelectedFeatures, featureIdxFile)
                    != ui->numSelectedFeatures) {
                    // Failed to read all indices
                    free(ui->selectedFeatureIndices);
                    ui->selectedFeatureIndices = NULL;
                    ui->numSelectedFeatures = 0;
                    printf("WARNING: Failed to read all selected feature indices\n");
                } else {
                    printf("Loaded %u selected feature indices\n", ui->numSelectedFeatures);
                }
            }
        }
        fclose(featureIdxFile);
    }

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

    // Set target FPS
    SetTargetFPS(60);

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

    // Free selected feature indices
    if (ui->selectedFeatureIndices != NULL) {
        free(ui->selectedFeatureIndices);
        ui->selectedFeatureIndices = NULL;
    }

    // Unload the render textures
    UnloadRenderTexture(canvasTexture);
    UnloadRenderTexture(processedCanvasTexture);

    // Unload the font if it's not the default
    if (gFont.texture.id != GetFontDefault().texture.id) {
        UnloadFont(gFont);
    }

    // Close raylib window
    CloseWindow();
}

// Render text with the loaded font and specified size
void renderText(int x, int y, int fontSize, const char *text, Color color) {
    DrawTextEx(gFont, text, (Vector2){(float)x, (float)y}, (float)fontSize, 2, color);
}

// Draw a button with rounded corners and hover effect
int drawButton(int x, int y, int w, int h, const char *text, Color baseColor, Color hoverColor, bool *isHovered) {
    Vector2 mousePos = GetMousePosition();
    Rectangle buttonRect = {(float)x, (float)y, (float)w, (float)h};
    *isHovered = CheckCollisionPointRec(mousePos, buttonRect);

    Color currentColor = *isHovered ? hoverColor : baseColor;

    // Draw rounded rectangle
    DrawRectangleRounded(buttonRect, 0.2, 10, currentColor);

    // Draw text centered within the button
    int textWidth = MeasureTextEx(gFont, text, 20, 2).x;
    int textX = x + (w - textWidth) / 2;
    int textY = y + (h - 20) / 2;
    renderText(textX, textY, 20, text, BLACK); // Use renderText for consistency

    return *isHovered; // Return whether the button is being hovered over
}


// Check if we should attempt prediction
int shouldPredict(void) {
    if (!canvasDirty)
        return 0;

    double currentTime = GetTime() * 1000; // Convert to milliseconds
    return (currentTime - lastDrawTime > PREDICTION_DELAY);
}
// Process mouse and keyboard events
int processEvents(DrawingUI *ui) {
    int wasDrawing = ui->drawing;

    // Check for window close
    if (WindowShouldClose()) {
        return 0;  // Exit
    }

    // Get mouse position
    Vector2 mousePos = GetMousePosition();

    // Check if mouse is pressed inside canvas
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        // Check if click is inside canvas
        if (mousePos.x >= CANVAS_X && mousePos.x < CANVAS_X + CANVAS_SIZE &&
            mousePos.y >= CANVAS_Y && mousePos.y < CANVAS_Y + CANVAS_SIZE) {

            ui->drawing = 1;

            // Reset prediction state when drawing starts
            if (!wasDrawing) {
                ui->prediction = -1;  // Clear prediction
                memset(ui->confidence, 0, sizeof(ui->confidence));
                ui->showProcessed = 0; // Hide processed view when drawing new character
            }

            // Convert mouse coordinates to canvas pixel coordinates
            int canvasX = (mousePos.x - CANVAS_X) * 28 / CANVAS_SIZE;
            int canvasY = (mousePos.y - CANVAS_Y) * 28 / CANVAS_SIZE;

             // Draw a circular "brush" centered on the pixel
            int brushRadius = 2; // Increased brush size
            for (int dy = -brushRadius; dy <= brushRadius; dy++) {
                for (int dx = -brushRadius; dx <= brushRadius; dx++) {
                    int px = canvasX + dx;
                    int py = canvasY + dy;
                    // Check if the pixel is within the brush radius
                    if (dx * dx + dy * dy <= brushRadius * brushRadius) {
                        if (px >= 0 && px < 28 && py >= 0 && py < 28) {
                            // Set pixel to maximum intensity (255) with falloff
                            float distance = sqrtf(dx * dx + dy * dy);
                            float intensity = 255 * (1.0f - distance / (float)brushRadius);
                            ui->canvas[py * 28 + px] = (uint8_t)fmax(ui->canvas[py * 28 + px], intensity);
                        }
                    }
                }
            }

            // Mark canvas as dirty and update last draw time
            canvasDirty = 1;
            lastDrawTime = GetTime() * 1000; // Convert to milliseconds
        }
    } else {
        ui->drawing = 0; // Mouse is not down
    }

        // Handle button clicks when mouse is released
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        int buttonY = CANVAS_Y + CANVAS_SIZE + 60;

        bool isHovered; // Dummy variable, not used
        // Check if click is on "Clear" button
        if (drawButton(CANVAS_X, buttonY, 120, 40, "Clear", LIGHTGRAY, (Color){200, 200, 200, 255}, &isHovered) && isHovered) {
            clearCanvas(ui);
        }

        // Check if click is on "Viz Mode" button
       if (drawButton(CANVAS_X + 130, buttonY, 160, 40, "Viz Mode", LIGHTGRAY, (Color){200, 200, 200, 255}, &isHovered) && isHovered) {
            cycleVisualizationMode(ui);
        }
    }

    // Handle key presses
    if (IsKeyPressed(KEY_T)) {
        // Toggle processed view
        ui->showProcessed = !ui->showProcessed;
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

    // Update button text based on new mode (no longer needed, handled in renderUI)
}

// Draw the canvas and UI elements
void renderUI(DrawingUI *ui) {
    // Begin frame drawing
    BeginDrawing();

    // Clear the background
    ClearBackground(RAYWHITE);

    // Draw panel divider
    DrawLine(PANEL_DIVIDER, 0, PANEL_DIVIDER, WINDOW_HEIGHT, LIGHTGRAY);

    // Draw title at the top (using renderText for consistent styling)
    DrawRectangle(0, 0, WINDOW_WIDTH, 40, (Color){230, 230, 230, 255});
    renderText(20, 10, 24, ui->showingLetters ? "Glyph identifier" : "Digit Recognizer", DARKGRAY);

    // === LEFT PANEL (Drawing Area) ===

    // Drawing instructions
    renderText(CANVAS_X, CANVAS_Y - 30, 20, "Draw in the box below", DARKGRAY);

    // Draw canvas background with drop shadow effect
    DrawRectangle(CANVAS_X + 4, CANVAS_Y + 4, CANVAS_SIZE, CANVAS_SIZE, GRAY);
    DrawRectangle(CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE, WHITE);

    // Draw canvas border
    DrawRectangleLines(CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE, BLACK);

    // Draw the pixels from our canvas data directly to the canvasTexture
    BeginTextureMode(canvasTexture);
    ClearBackground(WHITE); // Clear the texture before drawing
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            if (ui->canvas[y * 28 + x] > 0) {
                // Set color based on pixel intensity
                int intensity = ui->canvas[y * 28 + x];
                DrawPixel(x, y, (Color){0, 0, 0, intensity});
            }
        }
    }
    EndTextureMode();

    // Draw the canvasTexture scaled up
    DrawTexturePro(canvasTexture.texture,
                   (Rectangle){0, 0, 28, -28}, // Source rectangle (flip vertically)
                   (Rectangle){CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE}, // Destination
                   (Vector2){0, 0}, 0.0f, WHITE);


    // Display prediction timer below canvas
    if (canvasDirty && !ui->drawing) {
        double currentTime = GetTime() * 1000; // Convert to milliseconds
        double timeLeft = (lastDrawTime + PREDICTION_DELAY) - currentTime;
        char waitText[64];
        sprintf(waitText, "Predicting in %.1f sec...", timeLeft / 1000.0);
        renderText(CANVAS_X, CANVAS_Y + CANVAS_SIZE + 20, 18, waitText, BLUE);
    } else if (!canvasDirty) {
        renderText(CANVAS_X, CANVAS_Y + CANVAS_SIZE + 20, 18, "Predictions are automatic after drawing", DARKGRAY);
    }

    // Draw control buttons
    int buttonY = CANVAS_Y + CANVAS_SIZE + 60;

    bool isClearHovered, isVizModeHovered; // Variables to track hover state

    // Draw the "Clear" button with nicer styling and hover effect
    drawButton(CANVAS_X, buttonY, 120, 40, "Clear", LIGHTGRAY, (Color){200, 200, 200, 255}, &isClearHovered);


    // Draw visualization mode selection button with hover effect
      // Get the current visualization mode text
    const char* vizModeButtonText = "";
    switch (ui->vizMode) {
        case VIZ_MODE_NONE:        vizModeButtonText = "Mode: None";        break;
        case VIZ_MODE_PROCESSED:   vizModeButtonText = "Mode: Processed";   break;
        case VIZ_MODE_REFERENCE:   vizModeButtonText = "Mode: Reference";   break;
        case VIZ_MODE_HOG:         vizModeButtonText = "Mode: HOG";         break;
    }
    drawButton(CANVAS_X + 130, buttonY, 160, 40, vizModeButtonText, LIGHTGRAY, (Color){200, 200, 200, 255}, &isVizModeHovered);


    // Process visualization section
    int processedY = buttonY + 90; // Adjusted vertical position

    if (ui->vizMode == VIZ_MODE_PROCESSED && ui->prediction >= 0) {
        // Draw a header for the processed view
        renderText(CANVAS_X, processedY, 18, "Preprocessed Image:", DARKGRAY);
        processedY += 30;

        // Draw processed canvas with shadow effect
        DrawRectangle(CANVAS_X + 4, processedY + 4, CANVAS_SIZE / 1.5, CANVAS_SIZE / 1.5, GRAY);

        // Draw processed pixels directly onto the processedCanvasTexture
        BeginTextureMode(processedCanvasTexture);
        ClearBackground(WHITE); // Clear before drawing
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                if (ui->processedCanvas[y * 28 + x] > 0) {
                    int intensity = ui->processedCanvas[y * 28 + x];
                    DrawPixel(x, y, (Color){0, 0, 0, intensity});
                }
            }
        }
        EndTextureMode();

        // Draw the processedCanvasTexture scaled up
        DrawTexturePro(processedCanvasTexture.texture,
                       (Rectangle){0, 0, 28, -28},  // Flip vertically
                       (Rectangle){CANVAS_X, processedY, CANVAS_SIZE / 1.5, CANVAS_SIZE / 1.5},
                       (Vector2){0, 0}, 0.0f, WHITE);

        DrawRectangleLines(CANVAS_X, processedY, CANVAS_SIZE / 1.5, CANVAS_SIZE / 1.5, BLACK);
    }

    // === RIGHT PANEL (Results & Analysis) ===
    int rightPanelX = PANEL_DIVIDER + 30;
    int rightPanelY = 60;

    // If we have a prediction, show it
    if (ui->prediction >= 0) {
        char predText[100];
        char label = getLabelChar(ui->prediction, ui->showingLetters);

        // Define commonly confused pairs
        typedef struct {
            uint8_t class1;
            uint8_t class2;
            const char *pairName;
        } ConfusedPair;

        // Initialize with known confused pairs (in zero-based indices)
        ConfusedPair confusedPairs[] = {
            {8, 11, "i/l"},    // 'i' and 'l'
            {14, 20, "o/u"},   // 'o' and 'u'
            {2, 6, "c/g"}      // 'c' and 'g'
        };
        int numConfusedPairs = sizeof(confusedPairs) / sizeof(confusedPairs[0]);

        // Check if current prediction is part of a confused pair
        int isConfusedPair = 0;
        const char *confusedPairName = NULL;
        for (int i = 0; i < numConfusedPairs; i++) {
            if (ui->prediction == confusedPairs[i].class1 ||
                ui->prediction == confusedPairs[i].class2) {
                isConfusedPair = 1;
                confusedPairName = confusedPairs[i].pairName;
                break;
            }
        }

        // Draw a large prediction result
        sprintf(predText, "%c", label);
        renderText(rightPanelX, rightPanelY, 24, "Prediction:", DARKGRAY);

        // Draw a box around the prediction
        DrawRectangle(rightPanelX + 4, rightPanelY + 34, 76, 76, GRAY); // Shadow
        DrawRectangle(rightPanelX, rightPanelY + 30, 76, 76, WHITE);
        DrawRectangleLines(rightPanelX, rightPanelY + 30, 76, 76, BLACK);

        // Draw the predicted character in a large font
        renderText(rightPanelX + 20, rightPanelY + 40, 60, predText, isConfusedPair ? RED : BLUE);

        // Display confidence information
        sprintf(predText, "Confidence: %.2f%%", ui->confidence[ui->prediction] * 100.0);
        renderText(rightPanelX + 100, rightPanelY + 40, 20, predText, isConfusedPair ? RED : BLUE);

        // Add notice if this is a letter in a frequently confused pair
        if (isConfusedPair && ui->showingLetters) {
            sprintf(predText, "(Frequently confused pair: %s)", confusedPairName);
            renderText(rightPanelX + 100, rightPanelY + 70, 20, predText, RED);

            // Show info about the specialized classifier
            if (ui->specializedManager && ui->specializedManager->numClassifiers > 0) {
                renderText(rightPanelX + 100, rightPanelY + 100, 18, "Using specialized classifier", GREEN);
            }
        }

        // Show top 5 predictions with confidences
        int topInfoY = rightPanelY + 130;
        renderText(rightPanelX, topInfoY, 24, "Top Predictions:", DARKGRAY);
        DrawLine(rightPanelX, topInfoY + 30, rightPanelX + 300, topInfoY + 30, LIGHTGRAY);

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

        // Display top 5 with more visual appeal
        for (int i = 0; i < 5; i++) {
            if (topIndices[i] >= 0) {
                char topLabel = getLabelChar(topIndices[i], ui->showingLetters);

                // Check if this class is part of a confused pair
                int isConfused = 0;
                for (int j = 0; j < numConfusedPairs; j++) {
                    if (topIndices[i] == confusedPairs[j].class1 ||
                        topIndices[i] == confusedPairs[j].class2) {
                        isConfused = 1;
                        break;
                    }
                }

                // Highlight confused classes in rankings
                Color rankColor = isConfused ? RED : BLACK;
                Color barColor = isConfused ? RED : BLUE;

                // Character box
                DrawRectangle(rightPanelX, topInfoY + 40 + i*45, 40, 40, WHITE);
                DrawRectangleLines(rightPanelX, topInfoY + 40 + i*45, 40, 40, DARKGRAY);

                // Draw the character
                char charText[2] = {topLabel, '\0'};
                renderText(rightPanelX + 12, topInfoY + 45 + i*45, 30, charText, rankColor);

                // Draw confidence bar
                int barWidth = (int)(topConfidences[i] * 400.0);
                DrawRectangle(rightPanelX + 50, topInfoY + 50 + i*45, barWidth, 20,
                             ColorAlpha(barColor, 0.7f));

                // Draw confidence percentage
                char confText[20];
                sprintf(confText, "%.2f%%", topConfidences[i] * 100.0);
                renderText(rightPanelX + 60 + barWidth, topInfoY + 50 + i*45, 16, confText, rankColor);
            }
        }

        // === VISUALIZATION AREA ===
        int vizY = topInfoY + 280;

        switch (ui->vizMode) {
            case VIZ_MODE_NONE:
                // No visualization
                break;

            case VIZ_MODE_REFERENCE:
                // Show reference samples for the predicted letter
                if (gReferenceSamples.loaded) {
                    renderText(rightPanelX, vizY, 24, "Reference Samples:", DARKGRAY);
                    renderReferenceSamples(rightPanelX, vizY + 40, 400, 150, ui->prediction);
                } else {
                    renderText(rightPanelX, vizY, 24, "Reference Samples:", DARKGRAY);
                    renderText(rightPanelX, vizY + 40, 18, "Reference samples not available", RED);
                }
                break;

            case VIZ_MODE_HOG:
                // Calculate and show HOG feature visualization
                renderText(rightPanelX, vizY, 24, "HOG Feature Visualization:", DARKGRAY);

                if (ui->lastFeatures == NULL) {
                    // Allocate space for features if not already done
                    ui->lastFeatures = (double*)malloc(ui->model->numFeatures * sizeof(double));
                    if (ui->lastFeatures == NULL) {
                        renderText(rightPanelX, vizY + 40, 18, "Failed to allocate memory for feature visualization", RED);
                        break;
                    }

                    // We'll store features during processPrediction()
                    ui->lastFeaturesCount = ui->model->numFeatures;
                }

                // Visualize the HOG features
                if (ui->lastFeatures != NULL && gHOGViz.hasData) {
                    renderHOGVisualization(rightPanelX, vizY + 40, 250);
                } else {
                    renderText(rightPanelX, vizY + 40, 18, "HOG visualization not available", RED);
                    renderText(rightPanelX, vizY + 70, 18, "Draw a new letter to generate", DARKGRAY);
                }
                break;
        }
    } else {
        // No prediction yet - show instructions
        renderText(rightPanelX, rightPanelY, 24, "No prediction yet", DARKGRAY);
        renderText(rightPanelX, rightPanelY + 40, 18, "Draw a letter or digit in the canvas on the left.", DARKGRAY);
        renderText(rightPanelX, rightPanelY + 70, 18, "Prediction will be automatic.", DARKGRAY);


        renderText(rightPanelX, rightPanelY + 120, 18, "Visualization modes:", DARKGRAY);
        renderText(rightPanelX, rightPanelY + 150, 18, "• Processed: Preprocessed input", DARKGRAY);
        renderText(rightPanelX, rightPanelY + 180, 18, "• Reference: Training samples", DARKGRAY);
        renderText(rightPanelX, rightPanelY + 210, 18, "• HOG: HOG features", DARKGRAY);
    }

    // End the frame drawing
    EndDrawing();
}

// Clear the canvas
void clearCanvas(DrawingUI *ui) {
    // Clear the canvas data
    memset(ui->canvas, 0, 28*28);
    memset(ui->processedCanvas, 0, 28*28);  // Also clear the processed canvas

    // Clear the canvas texture
    BeginTextureMode(canvasTexture);
    ClearBackground(WHITE);
    EndTextureMode();

    BeginTextureMode(processedCanvasTexture);
    ClearBackground(WHITE);
    EndTextureMode();

    // Reset UI state
    ui->showProcessed = 0;                  // Hide the processed view
    ui->prediction = -1;                    // Clear prediction
    memset(ui->confidence, 0, sizeof(ui->confidence));
    canvasDirty = 0;                        // Canvas is clean
}

void preprocessCanvas(uint8_t *canvas, uint8_t *processedCanvas) {
    // Create preprocessing options
    PreprocessingOptions options;
    initDefaultPreprocessing(&options);

    // Less Aggressive Preprocessing:
    options.applyNormalization = 1;     // Keep size/position normalization
    options.applyThresholding = 1;      // Keep adaptive thresholding
    options.applySlantCorrection = 0;   // Disable slant correction (for now)
    options.applyNoiseRemoval = 1;      // Keep noise removal, but reduce aggressiveness
    options.applyStrokeNorm = 0;        // Disable stroke width normalization
    options.applyThinning = 0;          // Keep thinning disabled

    // Adjust parameters:
    options.borderSize = 2;             // Keep border padding
    options.targetStrokeWidth = 2;      // (Not used, since applyStrokeNorm is 0)
    options.noiseThreshold = 3;         // Increase noise threshold (less aggressive)

    // Use the full preprocessing pipeline from normalization.c
    preprocessImage(canvas, processedCanvas, 28, 28, &options);
}

void visualizeHOGFeatures(DrawingUI *ui, double *features, uint8_t predictedClass) {
    // Clear the visualization
    memset(&gHOGViz.featureMap, 0, sizeof(gHOGViz.featureMap));
    memset(&gHOGViz.cellStrengths, 0, sizeof(gHOGViz.cellStrengths));
    memcpy(gHOGViz.originalImage, ui->processedCanvas, 28*28); // Store original image
    gHOGViz.hasData = 0;  // Set to 0 initially, will set to 1 when successful

    // Early return if invalid inputs
    if (features == NULL || ui->model == NULL) {
        printf("Invalid inputs for HOG visualization\n");
        return;
    }

    // Get parameters
    int cellSize = CELL_SIZE;  // IMPORTANT: Must match training
    int numBins = NUM_BINS;
    int cellsX = 28 / cellSize;  // Number of cells horizontally
    int cellsY = 28 / cellSize;  // Number of cells vertically
    printf("visualizeHOGFeatures: cellSize=%d, numBins=%d, cellsX=%d, cellsY=%d\n", cellSize, numBins, cellsX, cellsY);

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

    // Map feature importance back to image pixels and store cell strengths
     for (int f = 0; f < ui->model->numFeatures; f++) {
        // Calculate which cell this feature belongs to
        int binIndex = f % numBins;
        int cellIndex = f / numBins;
        int cellY = cellIndex / cellsX;
        int cellX = cellIndex % cellsX;

        // IMPORTANT:  Add thorough bounds checking here.  This is a very common source of errors.
        if (cellY < 0 || cellY >= cellsY || cellX < 0 || cellX >= cellsX) {
            printf("ERROR: Invalid cell coordinates: cellY=%d, cellX=%d, cellsY=%d, cellsX=%d, f=%d, cellIndex=%d, binIndex=%d\n",
                   cellY, cellX, cellsY, cellsX, f, cellIndex, binIndex);
            continue; // Skip this feature if the coordinates are invalid.
        }
         // Store cell strength
        gHOGViz.cellStrengths[cellY][cellX][binIndex] = featureImportance[f];
         // For each pixel within the cell, accumulate feature importance
        for (int y = 0; y < cellSize; y++) {
            for (int x = 0; x < cellSize; x++) {
                int pixelY = cellY * cellSize + y;
                int pixelX = cellX * cellSize + x;

               if (pixelY < 0 || pixelY >= 28 || pixelX < 0 || pixelX >= 28) {
                    printf("ERROR: Invalid pixel coordinates: pixelY=%d, pixelX=%d, cellY=%d, cellX=%d, y=%d, x=%d\n",
                           pixelY, pixelX, cellY, cellX, y, x);
                    continue;  // Skip this pixel if coordinates are out of bounds
                }
                    // Accumulate scaled importance (for visualization)
                    double scaledImportance = featureImportance[f] * (1.0 + 0.2 * binIndex);
                    gHOGViz.featureMap[pixelY][pixelX] += scaledImportance;
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
// Improved HOG visualization function with feature selection awareness
void renderHOGVisualization(int x, int y, int size) {
    if (!gHOGViz.hasData) {
        // Draw placeholder if we don't have data
        DrawRectangle(x + 4, y + 4, size, size, GRAY); // Shadow
        DrawRectangle(x, y, size, size, LIGHTGRAY);
        DrawRectangleLines(x, y, size, size, DARKGRAY);
        renderText(x + 20, y + size/2 - 10, 18, "No HOG data available", DARKGRAY);
        return;
    }

    // Draw panel with shadow and border
    DrawRectangle(x + 4, y + 4, size, size, GRAY); // Shadow
    DrawRectangle(x, y, size, size, WHITE);

    // Draw explanation text
    renderText(x, y - 30, 16, "HOG Features (Histogram of Oriented Gradients)", DARKGRAY);
    renderText(x, y - 10, 14, "Arrows show dominant edge directions", DARKGRAY);

    // First, draw the original processed letter as a background
    for (int py = 0; py < 28; py++) {
        for (int px = 0; px < 28; px++) {
            if (gHOGViz.originalImage[py * 28 + px] > 50) {
                // Scale pixel coordinates to display size
                int dispX = x + px * size / 28;
                int dispY = y + py * size / 28;
                int pixSize = size / 28 + 1; // +1 to avoid gaps

                DrawRectangle(dispX, dispY, pixSize, pixSize, (Color){220, 220, 220, 255}); // Light gray
            }
        }
    }

    // Draw grid lines for cell boundaries
    int cellSize = CELL_SIZE;
    int cellsX = 28 / cellSize;
    int cellsY = 28 / cellSize;
    int numBins = NUM_BINS;

    // Grid lines
    for (int cy = 0; cy <= cellsY; cy++) { // Note: <= to draw the bottom/right borders
        int gridY = y + cy * cellSize * size / 28;
        DrawLine(x, gridY, x + size, gridY, ColorAlpha(LIGHTGRAY, 0.5f));
    }

    for (int cx = 0; cx <= cellsX; cx++) { // Note: <= to draw the bottom/right borders
        int gridX = x + cx * cellSize * size / 28;
        DrawLine(gridX, y, gridX, y + size, ColorAlpha(LIGHTGRAY, 0.5f));
    }

    // Draw HOG arrows with improved styling
    for (int cy = 0; cy < cellsY; cy++) {
        for (int cx = 0; cx < cellsX; cx++) {
            // Calculate cell center in display coordinates
            int centerX = x + (cx * cellSize + cellSize/2) * size / 28;
            int centerY = y + (cy * cellSize + cellSize/2) * size / 28;

            // Arrow length based on cell size
            int arrowLength = (size / 28) * cellSize / 2;

            // Get strongest orientations for this cell
            double maxMagnitude = 0.0;
            int dominantBins[3] = {-1, -1, -1};
            double dominantMags[3] = {0.0, 0.0, 0.0};

            // Find top 3 dominant orientations
            for (int bin = 0; bin < numBins; bin++) {
                double magnitude = gHOGViz.cellStrengths[cy][cx][bin];

                // Insert in sorted order
                for (int i = 0; i < 3; i++) {
                    if (magnitude > dominantMags[i]) {
                        // Shift down
                        for (int j = 2; j > i; j--) {
                            dominantBins[j] = dominantBins[j-1];
                            dominantMags[j] = dominantMags[j-1];
                        }
                        dominantBins[i] = bin;
                        dominantMags[i] = magnitude;
                        break;
                    }
                }

                if (magnitude > maxMagnitude) {
                    maxMagnitude = magnitude;
                }
            }

            // Draw a circle at the cell center
            DrawCircle(centerX, centerY, 2, ColorAlpha(GRAY, 0.5f));

            // Draw arrows for the dominant orientations
            for (int i = 0; i < 3; i++) {
                if (dominantBins[i] >= 0 && dominantMags[i] > 0.1) { // Only draw significant orientations
                    // Calculate orientation angle in radians
                    double angle = dominantBins[i] * M_PI / numBins;

                    // Calculate arrow length based on magnitude
                    float magRatio = dominantMags[i] / maxMagnitude;
                    int currentArrowLength = arrowLength * magRatio;

                    // Arrow start and end points
                    int startX = centerX - (int)(currentArrowLength * cos(angle));
                    int startY = centerY - (int)(currentArrowLength * sin(angle));
                    int endX = centerX + (int)(currentArrowLength * cos(angle));
                    int endY = centerY + (int)(currentArrowLength * sin(angle));

                    // Color gradient from red (strong) to blue (weak)
                    int r = (int)(255 * magRatio);
                    int g = (int)(100 * magRatio);
                    int b = (int)(255 * (1.0 - magRatio));
                    Color arrowColor = (Color){r, g, b, 255};

                    // Draw thicker arrow line for better visibility
                    float thickness = 2.0f * magRatio + 0.5f;
                    DrawLineEx((Vector2){startX, startY}, (Vector2){endX, endY}, thickness, arrowColor);

                    // Draw fancier arrowhead
                    double headAngle1 = angle + 3 * M_PI / 4;
                    double headAngle2 = angle - 3 * M_PI / 4;
                    int headLength = currentArrowLength / 3;

                    int head1X = endX - (int)(headLength * cos(headAngle1));
                    int head1Y = endY - (int)(headLength * sin(headAngle1));
                    int head2X = endX - (int)(headLength * cos(headAngle2));
                    int head2Y = endY - (int)(headLength * sin(headAngle2));

                    DrawLineEx((Vector2){endX, endY}, (Vector2){head1X, head1Y}, thickness, arrowColor);
                    DrawLineEx((Vector2){endX, endY}, (Vector2){head2X, head2Y}, thickness, arrowColor);
                }
            }
        }
    }

    // Draw border around the visualization
    DrawRectangleLines(x, y, size, size, BLACK);

    // Draw legend
    int legendX = x + size + 10;
    int legendY = y;
    renderText(legendX, legendY, 18, "Legend:", DARKGRAY);

    // Strong feature
    DrawLineEx((Vector2){legendX, legendY + 30}, (Vector2){legendX + 20, legendY + 30}, 2.0f, RED);
    renderText(legendX + 30, legendY + 25, 14, "Strong feature", DARKGRAY);

    // Medium feature
    DrawLineEx((Vector2){legendX, legendY + 50}, (Vector2){legendX + 20, legendY + 50}, 1.5f, PURPLE);
    renderText(legendX + 30, legendY + 45, 14, "Medium feature", DARKGRAY);

    // Weak feature
    DrawLineEx((Vector2){legendX, legendY + 70}, (Vector2){legendX + 20, legendY + 70}, 1.0f, BLUE);
    renderText(legendX + 30, legendY + 65, 14, "Weak feature", DARKGRAY);
}
// Display reference samples for comparison
void renderReferenceSamples(int x, int y, int width, int height, int letterIndex) {
    if (!gReferenceSamples.loaded || letterIndex < 0 || letterIndex >= 26) {
        return;
    }

    char title[64];
    sprintf(title, "Reference '%c' Samples", 'A' + letterIndex);
    renderText(x, y - 30, 16, "These are samples from the training dataset", DARKGRAY);
     // Draw the title centered above the samples
    renderText(x, y - 50, 20, title, DARKGRAY);


    // Calculate sample size and spacing
    int sampleSize = 100;
    int spacing = 20;

    // Draw each sample with stylish display
    for (int i = 0; i < gReferenceSamples.numSamplesPerClass; i++) {
        int sampleX = x + i * (sampleSize + spacing);

        // Draw sample with shadow effect
        DrawRectangle(sampleX + 4, y + 4, sampleSize, sampleSize, GRAY); // Shadow
        DrawRectangle(sampleX, y, sampleSize, sampleSize, WHITE);

        // Draw sample border
        DrawRectangleLines(sampleX, y, sampleSize, sampleSize, DARKGRAY);

        // Draw the sample pixels with antialised scaling
        for (int sy = 0; sy < 28; sy++) {
            for (int sx = 0; sx < 28; sx++) {
                if (gReferenceSamples.samples[letterIndex][i][sy * 28 + sx] > 50) {
                    int pixX = sampleX + sx * sampleSize / 28;
                    int pixY = y + sy * sampleSize / 28;
                    int pixSize = sampleSize / 28 + 1;

                    DrawRectangle(pixX, pixY, pixSize, pixSize, BLACK);
                }
            }
        }

        // Label the sample
        char numLabel[10];
        sprintf(numLabel, "Sample %d", i+1);
        DrawRectangle(sampleX, y + sampleSize + 5, sampleSize, 20, ColorAlpha(BLUE, 0.2f));
        renderText(sampleX + 5, y + sampleSize + 5, 16, numLabel, DARKGRAY);
    }

    // Draw an explanation about reference samples
    renderText(x, y + sampleSize + 35, 16, "These show how the model was trained.", DARKGRAY);
    renderText(x, y + sampleSize + 55, 16, "Compare your drawing to these.", DARKGRAY);
}
// Process the current drawing and make a prediction
void processPrediction(DrawingUI *ui) {
    memset(ui->processedCanvas, 0, 28*28);
    // Preprocess the canvas
    uint8_t processedCanvas[28*28];
    preprocessCanvas(ui->canvas, processedCanvas);

    // Get the correct cellSize - MUST match what was used in training
    int cellSize = CELL_SIZE;
    int numBins = NUM_BINS;

    // IMPORTANT: Calculate numFeatures the same way it was calculated during training
    int numFeatures = (28/cellSize) * (28/cellSize) * numBins;

    // Check if we're using feature selection
    int useFeatureSelection = (ui->selectedFeatureIndices != NULL && ui->numSelectedFeatures > 0);

    // Verify feature dimensions match the model - accounting for feature selection
    int expectedFeatures = useFeatureSelection ? ui->numSelectedFeatures : numFeatures;

    if (expectedFeatures != ui->model->numFeatures) {
        printf("ERROR: Feature dimension mismatch! Expected: %d, Got: %d\n",
               ui->model->numFeatures, expectedFeatures);
        if (useFeatureSelection) {
            printf("This might be due to a mismatch between the model and the loaded feature indices.\n");
        } else {
            printf("This is likely due to a cell size mismatch between training and prediction.\n");
        }
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

    // Create feature array for classification
    double *selectedFeatures = NULL;

    // If using feature selection, create a subset of features
    if (useFeatureSelection) {
        selectedFeatures = (double*)malloc(ui->numSelectedFeatures * sizeof(double));
        if (!selectedFeatures) {
            printf("Failed to allocate memory for selected features\n");
            free(hogFeatures.features);
            return;
        }

        // Extract the selected features
        for (uint32_t i = 0; i < ui->numSelectedFeatures; i++) {
            uint32_t selectedIdx = ui->selectedFeatureIndices[i];
            if (selectedIdx < hogFeatures.numFeatures) {
                selectedFeatures[i] = hogFeatures.features[selectedIdx];
            } else {
                selectedFeatures[i] = 0.0; // Safety fallback
            }
        }
    } else {
        // Use all features
        selectedFeatures = hogFeatures.features;
    }

    // Initialize prediction result
    PredictionResult result;
    int topN = ui->numClasses > 5 ? 5 : ui->numClasses; // Get top 5 or all if less than 5 classes

    // Perform classification based on whether we're using specialized classifiers
    if (ui->showingLetters && ui->specializedManager && ui->specializedManager->numClassifiers > 0) {
        // Use the two-stage classification approach for letters
        result = twoStageClassify(ui->model, ui->specializedManager, selectedFeatures, topN);
        printf("Using two-stage classification for letters\n");
    } else {
        // Use regular classification for digits or if no specialized classifiers
        result = predictNaiveBayesWithConfidence(ui->model, selectedFeatures, topN);
        printf("Using standard classification\n");
    }

    // Update UI with the result
    ui->prediction = result.prediction;

    // Set confidence scores for all classes
    memset(ui->confidence, 0, sizeof(ui->confidence));

    // Set confidence for the top classes we got back
    for (int i = 0; i < result.n; i++) {
        if (i < ui->numClasses) {
            uint8_t classIdx = result.topN[i];
            ui->confidence[classIdx] = result.classProbs[classIdx];
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
        // Determine how many features to visualize (all or selected)
        uint32_t numFeaturesToVisualize = hogFeatures.numFeatures;

        // Allocate/reallocate memory for features if needed
        if (ui->lastFeatures == NULL || ui->lastFeaturesCount != numFeaturesToVisualize) {
            // Free old memory if it exists
            if (ui->lastFeatures != NULL) {
                free(ui->lastFeatures);
            }

            // Allocate new memory
            ui->lastFeatures = (double*)malloc(numFeaturesToVisualize * sizeof(double));
            if (ui->lastFeatures != NULL) {
                ui->lastFeaturesCount = numFeaturesToVisualize;
            } else {
                ui->lastFeaturesCount = 0;
                printf("Failed to allocate memory for feature storage\n");
            }
        }

        // Copy features if memory allocation succeeded
        if (ui->lastFeatures != NULL) {
            // If using feature selection, create a subset of features for visualization
            if (ui->selectedFeatureIndices != NULL && ui->numSelectedFeatures > 0) {
                printf("Visualizing %u selected features\n", ui->numSelectedFeatures);
                // Just highlight the selected features for visualization
                for (uint32_t i = 0; i < numFeaturesToVisualize; i++) {
                    // Initialize all features to a very low value
                    ui->lastFeatures[i] = 0.01;
                }

                // Then boost the selected features
                for (uint32_t i = 0; i < ui->numSelectedFeatures; i++) {
                    uint32_t selectedIdx = ui->selectedFeatureIndices[i];
                    if (selectedIdx < numFeaturesToVisualize) {
                        ui->lastFeatures[selectedIdx] = hogFeatures.features[selectedIdx] * 2.0;
                    }
                }
            } else {
                // Just copy all features for visualization if not using feature selection
                memcpy(ui->lastFeatures, hogFeatures.features,
                    numFeaturesToVisualize * sizeof(double));
            }

            // Generate the HOG visualization
            visualizeHOGFeatures(ui, ui->lastFeatures, ui->prediction);
        }
    }

    // Free allocated memory in reverse order of allocation
    freePredictionResult(&result);
    if (useFeatureSelection && selectedFeatures) {
        free(selectedFeatures);
    }
    free(hogFeatures.features);

    // Add this right at the end of processPrediction() before returning
    // This makes sure we keep the processed view visible for better user feedback
    ui->showProcessed = 1;
}