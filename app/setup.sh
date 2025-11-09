#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Bat Face Recognition Project...${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    exit 1
fi

# Get the conda base directory
CONDA_BASE=$(conda info --base)

# Source conda initialization
echo -e "${YELLOW}Initializing conda...${NC}"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Remove existing environment if it exists
if conda env list | grep -q "bat_face_recognition"; then
    echo -e "${YELLOW}Removing existing bat_face_recognition environment...${NC}"
    conda deactivate 2>/dev/null || true
    conda env remove -n bat_face_recognition -y
fi

# Create new environment with Python 3.10
echo -e "${YELLOW}Creating conda environment: bat_face_recognition with Python 3.10${NC}"
conda create -n bat_face_recognition python=3.10 -y
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to create conda environment${NC}"
    exit 1
fi

# Activate the conda environment
echo -e "${YELLOW}Activating conda environment: bat_face_recognition${NC}"
eval "$(conda shell.bash hook)"
conda activate bat_face_recognition

# Verify Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo -e "${RED}Error: Python version is not 3.10 (current: $PYTHON_VERSION)${NC}"
    exit 1
fi

echo -e "${GREEN}Successfully activated conda environment: $CONDA_DEFAULT_ENV (Python $PYTHON_VERSION)${NC}"

# Install/update dependencies
echo -e "${YELLOW}Installing/updating dependencies...${NC}"
pip install -r requirements.txt

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to install dependencies${NC}"
    exit 1
fi

echo -e "${GREEN}Setup complete!${NC}" 