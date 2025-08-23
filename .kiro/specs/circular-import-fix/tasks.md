# Implementation Plan

Convert the configuration system from mixed data/model holders to pure data representations with separate model factory pattern, eliminating heavy imports during configuration loading.

- [x] 1. Refactor ImageModelConfig to pure data class
  - Remove ultralytics import from config/model_config.py
  - Update ImageModelConfig constructor to accept only data parameters (model_path, model_type, confidence_threshold)
  - Remove any model instance storage or handling
  - Update type hints to use strings instead of model instances
  - _Requirements: 1.1, 5.2_

- [ ] 2. Update ModelsConfig to create ImageModelConfig instances
  - Modify ModelsConfig._create_model_config to instantiate ImageModelConfig with data from config.yml
  - Update property methods (segmentation, pose, siamese) to return ImageModelConfig instances instead of raw dictionaries
  - Ensure no heavy imports are triggered during ModelsConfig initialization
  - _Requirements: 1.2, 4.1_

- [ ] 3. Create model factory package structure
  - Create app/models/ directory and __init__.py
  - Define ModelFactory protocol interface for type safety
  - Create basic package structure for model creation functions
  - _Requirements: 6.1_

- [ ] 4. Implement YOLO model factory functions
  - Create app/models/yolo_factory.py with create_yolo_model function
  - Implement heavy ultralytics import only within factory functions
  - Add proper error handling for missing files and import failures
  - Create specialized functions for segmentation and pose models with validation
  - _Requirements: 6.2, 6.4_

- [ ] 5. Update ImageTransforms to use model factory pattern
  - Modify ImageTransforms._get_model_from_config to use factory functions instead of direct model access
  - Update segment_image method to create models on-demand using factory
  - Update align_face_landmarks method to use factory for pose model creation
  - Ensure all model creation happens through factory functions, not config classes
  - _Requirements: 2.1, 2.2_

- [ ] 6. Update image_processor package imports
  - Simplify app/image_processor/__init__.py to remove circular dependency imports
  - Remove import of ImageModelConfig from package level
  - Update package documentation to reflect new import patterns
  - _Requirements: 3.1_

- [ ] 7. Update consuming modules to use direct imports
  - Update app/siamese_network/input_processor.py to import directly from transforms module
  - Update any other modules using package-level imports to use direct imports
  - Remove any code expecting loaded models from configuration classes
  - _Requirements: 7.2, 7.3_

- [ ] 8. Validate configuration loading performance
  - Test that configuration imports complete quickly (under 100ms)
  - Verify that no heavy imports occur during configuration loading
  - Confirm that model creation only happens when factory functions are called
  - _Requirements: 3.1, 3.2_

- [ ] 9. Update error handling for model creation
  - Implement proper exception classes for configuration and model creation errors
  - Add graceful error handling in factory functions for missing dependencies
  - Update consuming code to handle model creation failures appropriately
  - _Requirements: 6.4_

- [ ] 10. Clean up and validate all import chains
  - Remove any remaining heavy imports from config modules
  - Test all import paths to ensure no mutex lock errors occur
  - Verify that existing functionality works with new architecture
  - Update any remaining code that expects models from config classes
  - _Requirements: 1.1, 7.4_