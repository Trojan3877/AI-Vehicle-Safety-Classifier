#include "classifier.h"
#include <iostream>
#include <iomanip>

/*
 * This is the main driver file that:
 * - Loads vehicle data from a CSV file
 * - Applies two classification methods:
 *   1. Rule-based classification
 *   2. Score-based (simulated ML) classification
 * - Compares results against ground truth for accuracy
 */

int main() {
    // Load vehicle entries from CSV file
    auto data = Classifier::loadCSV("data/vehicles.csv");

    int correct_rule = 0, correct_score = 0;

    // Iterate through dataset and apply both classifiers
    for (const auto& vehicle : data) {
        bool pred_rule = Classifier::classifyRuleBased(vehicle);
        bool pred_score = Classifier::classifyScoreBased(vehicle);

        if (pred_rule == vehicle.is_safe) correct_rule++;
        if (pred_score == vehicle.is_safe) correct_score++;
    }

    // Output evaluation metrics
    std::cout << "Total Vehicles: " << data.size() << "\\n";
    std::cout << "Rule-Based Accuracy: " << std::fixed << std::setprecision(2)
              << (float(correct_rule) / data.size()) * 100 << "%\\n";
    std::cout << "Score-Based Accuracy: " << std::fixed << std::setprecision(2)
              << (float(correct_score) / data.size()) * 100 << "%\\n";

    return 0;
}
