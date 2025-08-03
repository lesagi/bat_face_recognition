import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import shutil
from datetime import datetime
import json
import importlib.util

# TensorFlow will be imported lazily when needed
import mlflow

# Add the parent directory to the Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config.general import *
from siamese_network.paths import *

# from background_replacement.paths import *  # TODO: Fix configuration system
from siamese_network.config import *
from background_replacement.config import *
from video_processing.config import *
from data_augmentation.config import *

# Visualization imports will be lazy-loaded when needed


# Initialize MLflow with configuration
def init_mlflow():
    """Initialize MLflow tracking URI based on configuration"""
    try:
        # Read config fresh from file to get latest values
        import importlib.util

        config_path = os.path.join(project_root, "config", "general.py")
        spec = importlib.util.spec_from_file_location("general_config", config_path)
        general_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(general_config)

        if general_config.MLFLOW_ENABLED:
            mlflow_uri = f"http://{general_config.MLFLOW_TRACKING_HOST}:{general_config.MLFLOW_TRACKING_PORT}"
            mlflow.set_tracking_uri(uri=mlflow_uri)
            print(f"MLflow tracking URI set to: {mlflow_uri}")
        else:
            print("MLflow tracking is disabled")
    except Exception as e:
        print(f"Warning: Could not initialize MLflow: {e}")
        # Fallback to global variables if config file reading fails
        try:
            if MLFLOW_ENABLED:
                mlflow_uri = f"http://{MLFLOW_TRACKING_HOST}:{MLFLOW_TRACKING_PORT}"
                mlflow.set_tracking_uri(uri=mlflow_uri)
                print(f"MLflow tracking URI set to (fallback): {mlflow_uri}")
        except:
            print("MLflow initialization failed completely")


# Initialize MLflow
init_mlflow()

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Change this to a secure secret key

# Ensure templates are reloaded in debug mode
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Configuration paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(project_root, "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "general.py")
BACKUP_DIR = os.path.join(CONFIG_DIR, "backups")

# Ensure directories exist
os.makedirs(BACKUP_DIR, exist_ok=True)


def backup_config():
    """Create a backup of the current config file with timestamp"""
    try:
        if not os.path.exists(CONFIG_FILE):
            print(f"Config file not found at: {CONFIG_FILE}")
            return False

        # Ensure backup directory exists
        try:
            os.makedirs(BACKUP_DIR, exist_ok=True)
        except Exception as e:
            print(f"Failed to create backup directory: {e}")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_DIR, f"config_{timestamp}.py")

        try:
            shutil.copy2(CONFIG_FILE, backup_file)
            print(f"Successfully created backup at: {backup_file}")
            return True
        except Exception as e:
            print(f"Failed to copy config file: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error in backup_config: {e}")
        return False


def load_config():
    # Get the absolute path to the config.py file
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config.py")
    )

    # Load the config module
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


def ensure_relative_path(path, base_path):
    """Convert a path to be relative to the base path if it's not already."""
    if path.startswith(base_path):
        return os.path.relpath(path, base_path)
    return path


def create_auto_backup(config_type, config_path):
    """Create an automatic backup of a config file with domain-specific naming"""
    try:
        if not os.path.exists(config_path):
            print(f"Config file not found at: {config_path}")
            return False

        # Create domain-specific backup directory
        backup_dir = os.path.join(BACKUP_DIR, config_type)
        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"{config_type}_{timestamp}.py")

        try:
            shutil.copy2(config_path, backup_file)
            print(f"Successfully created backup at: {backup_file}")
            return True
        except Exception as e:
            print(f"Failed to copy config file: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error in create_auto_backup: {e}")
        return False


def read_config_file(config_path):
    """Read all variables from a config file"""
    if not os.path.exists(config_path):
        return {}

    config_vars = {}
    with open(config_path, "r") as f:
        content = f.read()
        # Use ast to safely parse Python code
        import ast

        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try:
                            # Try to evaluate the value
                            value = ast.literal_eval(node.value)
                            config_vars[target.id] = value
                        except:
                            # If we can't evaluate, store the string representation
                            config_vars[target.id] = ast.unparse(node.value)
    return config_vars


def write_dict_to_file(f, dict_obj, indent=0):
    """Write a dictionary to a file with proper Python syntax"""
    indent_str = "    " * indent
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            f.write(f"{indent_str}{key} = {{\n")
            write_dict_to_file(f, value, indent + 1)
            f.write(f"{indent_str}}}\n")
        else:
            if isinstance(value, str):
                value = f'"{value}"'
            f.write(f"{indent_str}{key} = {value}\n")


@app.route("/config/save/<config_type>", methods=["POST"])
def save_config(config_type):
    try:
        data = request.json
        config_path = None
        config_dict = {}

        if config_type == "general":
            config_path = os.path.join(project_root, "config", "general.py")
            config_dict = {
                "GPU_ENABLED": data.get("GPU_ENABLED", False),
                "MLFLOW_ENABLED": data.get("MLFLOW_ENABLED", True),
                "MLFLOW_TRACKING_HOST": data.get("MLFLOW_TRACKING_HOST", "localhost"),
                "MLFLOW_TRACKING_PORT": data.get("MLFLOW_TRACKING_PORT", 8080),
            }
        elif config_type == "siamesePaths":
            config_path = os.path.join(project_root, "siamese_network", "paths.py")
            config_dict = {
                "BASE_DIR": data.get("BASE_DIR", ""),
                "TRAIN_DIR": data.get("TRAIN_DIR", ""),
                "TEST_DIR": data.get("TEST_DIR", ""),
                "MODEL_DIR": data.get("MODEL_DIR", ""),
                "LOG_DIR": data.get("LOG_DIR", ""),
            }
        elif config_type == "segmentationPaths":
            config_path = os.path.join(
                project_root, "background_replacement", "paths.py"
            )
            config_dict = {
                "BASE_DIR": data.get("BASE_DIR", ""),
                "TRAIN_DIR": data.get("TRAIN_DIR", ""),
                "TEST_DIR": data.get("TEST_DIR", ""),
                "MODEL_DIR": data.get("MODEL_DIR", ""),
                "LOG_DIR": data.get("LOG_DIR", ""),
            }
        elif config_type == "siamese":
            config_path = os.path.join(project_root, "siamese_network", "config.py")
            config_dict = {
                "IMAGE_SIZE": data.get("IMAGE_SIZE", 224),
                "BATCH_SIZE": data.get("BATCH_SIZE", 32),
                "EPOCHS": data.get("EPOCHS", 10),
                "LEARNING_RATE": data.get("LEARNING_RATE", 0.001),
                "MARGIN": data.get("MARGIN", 1.0),
            }
        elif config_type == "segmentation":
            config_path = os.path.join(
                project_root, "background_replacement", "config.py"
            )
            config_dict = {
                "IMAGE_SIZE": data.get("IMAGE_SIZE", 224),
                "BATCH_SIZE": data.get("BATCH_SIZE", 32),
                "EPOCHS": data.get("EPOCHS", 10),
                "LEARNING_RATE": data.get("LEARNING_RATE", 0.001),
            }
        elif config_type == "augmentationPaths":
            config_path = os.path.join(project_root, "data_augmentation", "paths.py")
            config_dict = {
                "BASE": data.get("AUGMENTATION_PATHS.BASE", ""),
                "INPUT": data.get("AUGMENTATION_PATHS.INPUT", ""),
                "OUTPUT": data.get("AUGMENTATION_PATHS.OUTPUT", ""),
            }
        elif config_type == "augmentation":
            config_path = os.path.join(project_root, "data_augmentation", "config.py")
            # Handle AUGMENTATION_TYPES as a list
            augmentation_types = data.get("AUGMENTATION_TYPES", [])
            print(f"Received augmentation types: {augmentation_types}")  # Debug print

            if isinstance(augmentation_types, str):
                augmentation_types = [augmentation_types] if augmentation_types else []
            elif not isinstance(augmentation_types, list):
                augmentation_types = []

            config_dict = {
                "AUGMENTATION_TYPES": augmentation_types,
                "AUGMENTATION_FACTOR": data.get("AUGMENTATION_FACTOR", 1),
            }
        elif config_type == "video":
            config_path = os.path.join(project_root, "video_processing", "config.py")
            config_dict = {
                "FRAME_EXTRACTION_RATE": data.get("FRAME_EXTRACTION_RATE", 1),
                "MIN_FACE_SIZE": data.get("MIN_FACE_SIZE", 100),
                "MAX_FACE_SIZE": data.get("MAX_FACE_SIZE", 1000),
                "CONFIDENCE_THRESHOLD": data.get("CONFIDENCE_THRESHOLD", 0.5),
            }
        elif config_type == "visualizationConfig":
            config_path = os.path.join(project_root, "visualization", "config.py")
            config_dict = {
                "MODEL_PATH": data.get("modelPath", ""),
                "INPUT_DIR": data.get("inputDir", ""),
                "OUTPUT_DIR": data.get("outputDir", ""),
                "NESTING": data.get("nesting", None),
                "SAMPLE_SIZE": data.get("sampleSize", 10),
                "FIGURE_SIZE": (
                    data.get("figureWidth", 15),
                    data.get("figureHeight", 3),
                ),
                "ROW_SPACING": data.get("rowSpacing", 0.5),
                "IMAGE_SIZE": (
                    data.get("imageWidth", 224),
                    data.get("imageHeight", 224),
                ),
            }
        else:
            return jsonify(
                {"success": False, "error": f"Invalid config type: {config_type}"}
            )

        if config_path and config_dict:
            # Read existing config file
            existing_config = read_config_file(config_path)

            # Update only the changed values
            for key, value in config_dict.items():
                existing_config[key] = value

            # Write the updated config back to file
            with open(config_path, "w") as f:
                f.write('"""Configuration settings."""\n\n')
                f.write("import os\n\n")
                f.write("# Base directory path\n")
                f.write(
                    "BASE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n\n"
                )

                # Write the configuration
                for key, value in existing_config.items():
                    if isinstance(value, dict):
                        f.write(f"{key} = {{\n")
                        write_dict_to_file(f, value, 1)
                        f.write("}\n\n")
                    else:
                        if isinstance(value, str):
                            value = f'"{value}"'
                        f.write(f"{key} = {value}\n")

            # Create backup
            create_auto_backup(config_type, config_path)

            # Re-initialize MLflow if general config was updated
            if config_type == "general":
                init_mlflow()

            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Invalid config type"})

    except Exception as e:
        print(f"Error saving config: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/")
def index():
    # Load visualization config
    visualization_config_dict = load_visualization_config()

    # Combine all configs into a single dictionary
    config = {
        "GPU_ENABLED": GPU_ENABLED,
        "MLFLOW_ENABLED": MLFLOW_ENABLED,
        "MLFLOW_TRACKING_HOST": MLFLOW_TRACKING_HOST,
        "MLFLOW_TRACKING_PORT": MLFLOW_TRACKING_PORT,
        "SIAMESE_PATHS": SIAMESE_PATHS,
        # "SEGMENTATION_PATHS": SEGMENTATION_PATHS,  # TODO: Fix configuration system
        "INPUT_EDGE_LENGTH": INPUT_EDGE_LENGTH,
        "EPOCHS": EPOCHS,
        "TRAINING_DATA_MAX_SIZE_LIMIT": TRAINING_DATA_MAX_SIZE_LIMIT,
        "VIDEOS_DIR": VIDEOS_DIR,
        "VIDEOS_PROCESSING_LIMIT": VIDEOS_PROCESSING_LIMIT,
        "AUGMENTATION_PATHS": AUGMENTATION_PATHS,
        "AUGMENTATION_FACTOR": AUGMENTATION_FACTOR,
        "VISUALIZATION_CONFIG": visualization_config_dict,
    }
    return render_template("index.html", config=config)


def load_visualization_config():
    """Load visualization config from file"""
    visualization_config_path = os.path.join(project_root, "visualization", "config.py")
    if os.path.exists(visualization_config_path):
        # Import the config module directly
        spec = importlib.util.spec_from_file_location(
            "visualization_config", visualization_config_path
        )
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        # Create a dictionary with the config values
        visualization_config_dict = {
            "MODEL_PATH": config.MODEL_PATH,
            "INPUT_DIR": config.INPUT_DIR,
            "OUTPUT_DIR": config.OUTPUT_DIR,
            "NESTING": config.NESTING,
            "SAMPLE_SIZE": int(config.SAMPLE_SIZE),
            "FIGURE_SIZE": tuple(float(x) for x in config.FIGURE_SIZE),
            "ROW_SPACING": float(config.ROW_SPACING),
            "IMAGE_SIZE": tuple(int(x) for x in config.IMAGE_SIZE),
        }

        return visualization_config_dict
    else:
        return {
            "MODEL_PATH": "",
            "INPUT_DIR": "",
            "OUTPUT_DIR": "",
            "NESTING": "",
            "SAMPLE_SIZE": 10,
            "FIGURE_SIZE": (8.27, 5),
            "ROW_SPACING": 0.5,
            "IMAGE_SIZE": (350, 350),
        }


@app.route("/config")
def config_editor():
    try:
        # Load visualization config
        visualization_config_dict = load_visualization_config()

        # Combine all configs into a single dictionary
        config = {
            "GPU_ENABLED": GPU_ENABLED,
            "MLFLOW_ENABLED": MLFLOW_ENABLED,
            "MLFLOW_TRACKING_HOST": MLFLOW_TRACKING_HOST,
            "MLFLOW_TRACKING_PORT": MLFLOW_TRACKING_PORT,
            "SIAMESE_PATHS": SIAMESE_PATHS,
            # "SEGMENTATION_PATHS": SEGMENTATION_PATHS,  # TODO: Fix configuration system
            "INPUT_EDGE_LENGTH": INPUT_EDGE_LENGTH,
            "EPOCHS": EPOCHS,
            "TRAINING_DATA_MAX_SIZE_LIMIT": TRAINING_DATA_MAX_SIZE_LIMIT,
            "VIDEOS_DIR": VIDEOS_DIR,
            "VIDEOS_PROCESSING_LIMIT": VIDEOS_PROCESSING_LIMIT,
            "AUGMENTATION_PATHS": AUGMENTATION_PATHS,
            "AUGMENTATION_FACTOR": AUGMENTATION_FACTOR,
            "VISUALIZATION_CONFIG": visualization_config_dict,
        }

        return render_template("config.html", config=config)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        import traceback

        traceback.print_exc()
        return render_template("config.html", config={})


@app.route("/backups")
def list_backups():
    backups = {}
    if os.path.exists(BACKUP_DIR):
        for domain in os.listdir(BACKUP_DIR):
            domain_path = os.path.join(BACKUP_DIR, domain)
            if os.path.isdir(domain_path):
                domain_backups = []
                for file in os.listdir(domain_path):
                    if file.endswith(".py"):
                        file_path = os.path.join(domain_path, file)
                        timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                        domain_backups.append(
                            {
                                "id": os.path.join(domain, file),
                                "filename": file,
                                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "size": os.path.getsize(file_path),
                            }
                        )
                if domain_backups:
                    backups[domain] = sorted(
                        domain_backups, key=lambda x: x["timestamp"], reverse=True
                    )
    return render_template("backups.html", backups=backups)


@app.route("/create_backup", methods=["POST"])
def create_backup():
    try:
        # Create a backup of the current config
        if backup_config():
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Failed to create backup"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/get_backup/<path:filename>")
def get_backup(filename):
    try:
        backup_path = os.path.join(BACKUP_DIR, filename)
        if os.path.exists(backup_path):
            with open(backup_path, "r") as f:
                content = f.read()
            return jsonify({"success": True, "content": content})
        else:
            return jsonify(
                {"success": False, "error": f"Backup file not found at: {backup_path}"}
            )
    except Exception as e:
        print(f"Error reading backup file: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/restore_backup/<path:filename>", methods=["POST"])
def restore_backup(filename):
    try:
        backup_path = os.path.join(BACKUP_DIR, filename)
        if os.path.exists(backup_path):
            # Determine the original config file path based on the domain
            domain = filename.split("/")[0]
            if domain == "general":
                config_path = os.path.join(project_root, "config", "general.py")
            elif domain == "siamese_paths":
                config_path = os.path.join(project_root, "siamese_network", "paths.py")
            elif domain == "segmentation_paths":
                config_path = os.path.join(
                    project_root, "background_replacement", "paths.py"
                )
            elif domain == "siamese_config":
                config_path = os.path.join(project_root, "siamese_network", "config.py")
            elif domain == "segmentation_config":
                config_path = os.path.join(
                    project_root, "background_replacement", "config.py"
                )
            elif domain == "augmentation_config":
                config_path = os.path.join(
                    project_root, "data_augmentation", "config.py"
                )
            elif domain == "video_config":
                config_path = os.path.join(
                    project_root, "video_processing", "config.py"
                )
            else:
                return jsonify({"success": False, "error": "Invalid backup domain"})

            # Create a backup of the current config before restoring
            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pre_restore_backup = os.path.join(
                BACKUP_DIR, domain, f"pre_restore_{current_timestamp}.py"
            )
            if os.path.exists(config_path):
                shutil.copy2(config_path, pre_restore_backup)

            # Restore from backup
            shutil.copy2(backup_path, config_path)
            return jsonify({"success": True})
        else:
            return jsonify(
                {"success": False, "error": f"Backup file not found at: {backup_path}"}
            )
    except Exception as e:
        print(f"Error restoring backup: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/delete_backup/<path:filename>", methods=["DELETE"])
def delete_backup(filename):
    try:
        backup_path = os.path.join(BACKUP_DIR, filename)
        if os.path.exists(backup_path):
            os.remove(backup_path)
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Backup file not found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/config/save/saliency", methods=["POST"])
def save_saliency_config():
    try:
        data = request.get_json()

        config_content = f'''"""
Configuration for saliency maps generation.
"""

# Model configuration
MODEL_PATH = "{data['modelPath']}"  # Path to the trained model

# Input/Output configuration
INPUT_DIR = "{data['inputDir']}"  # Directory containing input images
OUTPUT_DIR = "{data['outputDir']}"  # Directory for saving saliency maps
NESTING = {data['nesting']}  # Optional subdirectory structure

# Image processing configuration
SAMPLE_SIZE = {data['sampleSize']}  # Number of images to process for saliency maps
FIGURE_SIZE = ({data['figureSize'][0]}, {data['figureSize'][1]})  # Size of the output figure (width, height per image)
ROW_SPACING = {data['rowSpacing']}  # Space between rows in the figure
IMAGE_SIZE = ({data['imageSize'][0]}, {data['imageSize'][1]})  # Size to resize images for display
'''

        config_path = os.path.join(project_root, "visualization", "config.py")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w") as f:
            f.write(config_content)

        # Create backup
        create_auto_backup(config_path, "visualization")

        return jsonify({"message": "Saliency maps configuration saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/visualize", methods=["POST"])
def visualize():
    try:
        # Lazy import TensorFlow and visualization module
        import tensorflow as tf
        from visualization.visualize import visualize_saliency_maps

        # Load visualization config
        visualization_config = load_visualization_config()

        # Create timestamped output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = os.path.join(
            visualization_config["OUTPUT_DIR"], f"saliency_maps_{timestamp}.pdf"
        )

        # Ensure output directory exists
        os.makedirs(visualization_config["OUTPUT_DIR"], exist_ok=True)

        # Call visualization function with config values
        visualize_saliency_maps(
            model_path=visualization_config["MODEL_PATH"],
            input_dir=visualization_config["INPUT_DIR"],
            output_pdf=output_pdf,
            nesting=visualization_config["NESTING"],
            sample_size=visualization_config["SAMPLE_SIZE"],
            figure_size=visualization_config["FIGURE_SIZE"],
            row_spacing=visualization_config["ROW_SPACING"],
            image_size=visualization_config["IMAGE_SIZE"],
        )

        return jsonify({"success": True, "output_path": output_pdf})

    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/create_saliency_maps", methods=["POST"])
def create_saliency_maps():
    try:
        # Load visualization config
        visualization_config = load_visualization_config()

        # Get config values
        model_path = visualization_config["MODEL_PATH"]
        input_dir = visualization_config["INPUT_DIR"]
        output_dir = visualization_config["OUTPUT_DIR"]
        nesting = visualization_config["NESTING"]

        if not model_path:
            raise ValueError("Model path is required")
        if not input_dir:
            raise ValueError("Input directory is required")
        if not output_dir:
            raise ValueError("Output directory is required")

        # Only use nesting for output directory
        effective_output_dir = (
            os.path.join(output_dir, nesting) if nesting else output_dir
        )

        print(f"Using input directory: {input_dir}")
        print(f"Using output directory: {effective_output_dir}")

        # Lazy import TensorFlow and visualization module
        import tensorflow as tf
        from visualization.saliency import SiameseModelSaliencyMapCreator

        # Load the model directly
        model = tf.saved_model.load(model_path)

        # Create saliency maps
        creator = SiameseModelSaliencyMapCreator(
            model=model,
            input_dir_path=input_dir,
            output_dir_path=effective_output_dir,
            nesting=None,  # We've already handled the nesting above
        )
        creator.compute_saliency_map()

        return jsonify(
            {
                "status": "success",
                "message": "Saliency maps created successfully",
                "output_path": creator.output_file_path,
            }
        )
    except Exception as e:
        print(f"Error creating saliency maps: {str(e)}")
        import traceback

        traceback.print_exc()  # This will print the full error traceback
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/browse_folders")
def browse_folders():
    requested_path = request.args.get("path", str(Path.home()))

    try:
        path = Path(requested_path).resolve()

        # Security check - ensure we're not going outside allowed areas
        # You might want to restrict this further based on your needs
        if not str(path).startswith("/Users/"):
            path = Path.home()

        items = []
        if path.exists() and path.is_dir():
            try:
                for item in sorted(path.iterdir()):
                    if item.is_dir() and not item.name.startswith("."):
                        items.append(
                            {"name": item.name, "path": str(item), "type": "folder"}
                        )
            except PermissionError:
                pass  # Skip folders we can't read

        parent_dir = str(path.parent) if path.parent != path else None

        return jsonify(
            {"current_path": str(path), "parent_dir": parent_dir, "items": items}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/save_segmentation_paths", methods=["POST"])
def save_segmentation_paths():
    paths = {
        "BASE": request.form.get("SEGMENTATION_PATHS.BASE"),
        "DATA": request.form.get("SEGMENTATION_PATHS.DATA"),
        "MODEL": request.form.get("SEGMENTATION_PATHS.MODEL"),
        "CONFIG": request.form.get("SEGMENTATION_PATHS.CONFIG"),
    }

    # Validate paths exist
    invalid_paths = []
    for key, path in paths.items():
        if path and not os.path.exists(path):
            invalid_paths.append(f"{key}: {path}")

    if invalid_paths:
        return (
            jsonify(
                {
                    "error": "The following paths do not exist: "
                    + ", ".join(invalid_paths)
                }
            ),
            400,
        )

    # Save to your config (replace with your actual config saving logic)
    try:
        # Example: save to a JSON config file
        # with open('config.json', 'w') as f:
        #     json.dump(paths, f)

        return jsonify({"success": True, "message": "Paths saved successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to save paths: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
