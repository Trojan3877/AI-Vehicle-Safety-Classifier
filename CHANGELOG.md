# Changelog

## [Unreleased]
- Added `docker-compose.yml` to run classifier and optional dashboard.
- Introduced Python unit tests for preprocessing utilities.

## [2025-06-12] v1.0.1
- Fixed memory leak in C++ classifier (Issue #23).
- Improved input validation in `predict.py`.

## [2025-05-30] v1.0.0
- Initial release of AI Vehicle Safety Classifier.
- Includes:
  - C++ core classifier (`src/`).
  - Python mirror script (`predict.py`).
  - Modular code, sample images, and evaluation metrics.
