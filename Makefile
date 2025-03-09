CC = gcc
CFLAGS = -Wall -Wextra -g

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Get all source files
ALL_SRCS = $(wildcard $(SRC_DIR)/*.c)

# Normal classifier (exclude the interactive main and UI components)
CLASSIFIER_SRCS = $(filter-out $(SRC_DIR)/main_interactive.c $(SRC_DIR)/ui_drawer.c, $(ALL_SRCS))
CLASSIFIER_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(CLASSIFIER_SRCS))
CLASSIFIER_EXEC = $(BIN_DIR)/mnist_classifier

# Interactive app (exclude the regular main)
INTERACTIVE_SRCS = $(filter-out $(SRC_DIR)/main.c, $(ALL_SRCS))
INTERACTIVE_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(INTERACTIVE_SRCS))
INTERACTIVE_EXEC = $(BIN_DIR)/interactive_recognizer

# SDL flags for the interactive app
SDL_FLAGS = -lSDL2 -lSDL2_ttf

# Default target builds both
all: directories classifier interactive

# Just build the classifier
classifier: directories $(CLASSIFIER_EXEC)

# Just build the interactive app
interactive: directories $(INTERACTIVE_EXEC)

directories:
	mkdir -p $(OBJ_DIR) $(BIN_DIR)

# Regular classifier
$(CLASSIFIER_EXEC): $(CLASSIFIER_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Interactive app with SDL
$(INTERACTIVE_EXEC): $(INTERACTIVE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm $(SDL_FLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all classifier interactive clean directories