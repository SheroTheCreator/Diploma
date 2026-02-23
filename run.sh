#!/bin/bash
echo "==================================================="
echo "  Starting Portfolio Optimization Tool..."
echo "==================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "ERROR: python3 could not be found."
    echo "Please install Python 3.9 or newer."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment (venv)..."
    python3 -m venv venv
fi

# Activate venv
echo "[2/3] Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "[3/3] Checking and installing dependencies..."
pip install -q -r requirements.txt

# Run the app
echo ""
echo "==================================================="
echo "  Launching Web Application in your browser..."
echo "  (Keep this terminal open while using the app)"
echo "==================================================="
streamlit run app.py