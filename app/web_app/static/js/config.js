// File and directory selection functions
function selectFile(inputId, accept) {
    const input = document.getElementById(inputId);
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    if (accept) {
        fileInput.accept = accept;
    }
    fileInput.onchange = function() {
        if (this.files && this.files.length > 0) {
            input.value = this.files[0].name;
        }
    };
    fileInput.click();
}

function selectDirectory(inputId) {
    const input = document.getElementById(inputId);
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.webkitdirectory = true;  // Enable directory selection
    fileInput.directory = true;        // For Firefox
    fileInput.multiple = true;         // Required for directory selection
    
    fileInput.onchange = function() {
        if (this.files && this.files.length > 0) {
            // Get the directory path from the first file
            const path = this.files[0].webkitRelativePath;
            const directory = path.split('/')[0];
            input.value = directory;
        }
    };
    fileInput.click();
}

// Add event listeners for directory pickers
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.directory-picker').forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            selectDirectory(targetId);
        });
    });
});

document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        const config = {};
        
        // Special handling for visualization form
        if (this.id === 'visualizationConfigForm') {
            config['modelPath'] = document.getElementById('visModelPath').value;
            config['inputDir'] = document.getElementById('visInputDir').value;
            config['outputDir'] = document.getElementById('visOutputDir').value;
            config['nesting'] = document.getElementById('visNesting').value;
            config['sampleSize'] = parseInt(document.getElementById('visSampleSize').value);
            config['figureWidth'] = parseFloat(document.getElementById('visFigureWidth').value);
            config['figureHeight'] = parseFloat(document.getElementById('visFigureHeight').value);
            config['rowSpacing'] = parseFloat(document.getElementById('visRowSpacing').value);
            config['imageWidth'] = parseInt(document.getElementById('visImageWidth').value);
            config['imageHeight'] = parseInt(document.getElementById('visImageHeight').value);
        } else {
            // Handle other form fields
            for (let [key, value] of formData.entries()) {
                if (key === 'sampleSize' || key === 'figureWidth' || key === 'figureHeight' || 
                    key === 'rowSpacing' || key === 'imageWidth' || key === 'imageHeight') {
                    config[key] = parseInt(value);
                } else {
                    config[key] = value;
                }
            }
        }
        
        // Map form IDs to config types
        const configTypeMap = {
            'siamesePathsForm': 'siamesePaths',
            'siameseForm': 'siamese',
            'segmentationPathsForm': 'segmentationPaths',
            'segmentationForm': 'segmentation',
            'generalForm': 'general',
            'videoForm': 'video',
            'visualizationConfigForm': 'visualizationConfig'
        };
        
        const configType = configTypeMap[this.id];
        if (!configType) {
            showErrorMessage('Invalid form ID: ' + this.id);
            return;
        }
        
        fetch(`/config/save/${configType}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(config)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showSuccessMessage('Configuration saved successfully');
            } else {
                showErrorMessage(data.error || 'Failed to save configuration');
            }
        })
        .catch(error => {
            showErrorMessage('Error saving configuration: ' + error);
        });
    });
});

// Add event listener for visualization form
document.getElementById('visualizationConfigForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = {
        modelPath: document.getElementById('visModelPath').value,
        inputDir: document.getElementById('visInputDir').value,
        outputDir: document.getElementById('visOutputDir').value,
        nesting: document.getElementById('visNesting').value,
        sampleSize: parseInt(document.getElementById('visSampleSize').value),
        figureWidth: parseFloat(document.getElementById('visFigureWidth').value),
        figureHeight: parseFloat(document.getElementById('visFigureHeight').value),
        rowSpacing: parseFloat(document.getElementById('visRowSpacing').value),
        imageWidth: parseInt(document.getElementById('visImageWidth').value),
        imageHeight: parseInt(document.getElementById('visImageHeight').value)
    };
    
    try {
        const response = await fetch('/config/save/visualizationConfig', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('Configuration saved successfully!', 'success');
        } else {
            showNotification('Error saving configuration: ' + result.error, 'error');
        }
    } catch (error) {
        showNotification('Error saving configuration: ' + error.message, 'error');
    }
});

// Function to run visualization
async function runVisualization() {
    try {
        const response = await fetch('/visualize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('Visualization completed successfully! PDF saved to: ' + result.output_path, 'success');
        } else {
            showNotification('Error during visualization: ' + result.error, 'error');
        }
    } catch (error) {
        showNotification('Error during visualization: ' + error.message, 'error');
    }
} 