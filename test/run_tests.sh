#!/bin/bash
# Test runner script for FVC project

set -e

echo "=========================================="
echo "Running FVC Unit Tests"
echo "=========================================="

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing..."
    pip install pytest pytest-cov
fi

# Run tests with coverage
echo ""
echo "Running tests with coverage..."
pytest test/ -v --cov=lib --cov-report=html --cov-report=term-missing

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Coverage report generated in htmlcov/index.html"

