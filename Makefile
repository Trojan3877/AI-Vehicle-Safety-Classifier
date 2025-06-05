# Makefile for AI Vehicle Safety Classifier

CXX := g++
CXXFLAGS := -std=c++17 -Wall -O2

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin

CPP_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_FILES))
EXEC := $(BIN_DIR)/vehicle_safety_classifier

.PHONY: all clean test run

all: $(EXEC)

# Build the C++ binary
$(EXEC): $(OBJ_FILES) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) results/ models/

# Run Python tests (if you have tests written in Python)
test:
	pytest --maxfail=1 --disable-warnings -q

# Example run command: adjust parameters as needed
run: all
	$(EXEC) --config config/dev.yaml

