#!/bin/bash

# Script to run the Flask app with the correct conda Python interpreter
echo "Starting Flask app with conda Python..."
echo "App will be available at: http://localhost:5001"
echo "Press Ctrl+C to stop the server"
echo ""

# Use the explicit conda Python path
/Users/sagilevi/miniconda3/envs/bat_face_recognition/bin/python app.py 