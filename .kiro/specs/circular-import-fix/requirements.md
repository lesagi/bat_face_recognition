# Config Class Refactoring - Requirements Document

## Introduction

The project is experiencing a critical import chain bottleneck issue that manifests as mutex lock failures during module initialization. The root cause is that configuration classes are importing heavy dependencies (`ultralytics.YOLO`) instead of simply representing the configuration data from `config.yml`.

Configuration classes should be pure data representations that only hold configuration values (strings, numbers, booleans) without importing or instantiating any heavy dependencies. Model instantiation should happen separately when the models are actually needed.

## Requirements

### Requirement 1: Pure Data Configuration Classes

**User Story:** As a developer, I want configuration classes to only represent data from config.yml without importing heavy dependencies, so that configuration loading is fast and doesn't cause import issues.

#### Acceptance Criteria

1. WHEN importing any configuration class THEN it SHALL complete instantly without loading heavy dependencies
2. WHEN accessing configuration properties THEN they SHALL return simple data types (string, number, boolean, dict) or other Config classes
3. WHEN configuration classes are instantiated THEN they SHALL only hold data from config.yml
4. IF configuration data is invalid THEN the system SHALL provide clear validation errors without attempting to load models

### Requirement 2: Separate Model Instantiation from Configuration

**User Story:** As a developer, I want model instantiation to be separate from configuration loading, so that I can load config quickly and create models only when needed.

#### Acceptance Criteria

1. WHEN loading configuration THEN no models SHALL be instantiated automatically
2. WHEN I need to create a model THEN I SHALL use the configuration data to instantiate it separately
3. WHEN configuration contains model paths THEN they SHALL be stored as strings without validation of file existence
4. IF model instantiation fails THEN it SHALL not affect configuration loading

### Requirement 3: Fast Configuration Loading

**User Story:** As a developer, I want configuration loading to be fast and lightweight, so that application startup time is minimized.

#### Acceptance Criteria

1. WHEN importing configuration modules THEN they SHALL complete in under 100ms
2. WHEN loading config.yml THEN only YAML parsing SHALL occur, no model loading
3. WHEN accessing configuration properties THEN they SHALL return immediately without computation
4. IF config.yml is malformed THEN clear parsing errors SHALL be provided without attempting model operations

### Requirement 4: Consistent Configuration Class Pattern

**User Story:** As a developer, I want all configuration classes to follow the same pattern of being pure data holders, so that the system is consistent and maintainable.

#### Acceptance Criteria

1. WHEN creating new configuration classes THEN they SHALL only contain properties that map to config.yml data
2. WHEN configuration classes are defined THEN they SHALL not import any heavy dependencies
3. WHEN adding new configuration sections THEN they SHALL follow the same data-only pattern
4. IF a configuration class needs to create objects THEN it SHALL provide factory methods that are called separately

### Requirement 5: Simple and Clear Configuration API

**User Story:** As a developer, I want configuration classes to have a simple, predictable API that matches the config.yml structure, so that they are easy to use and understand.

#### Acceptance Criteria

1. WHEN accessing configuration properties THEN they SHALL directly correspond to config.yml keys
2. WHEN using ImageModelConfig THEN it SHALL have exactly three properties: model_path (string), model_type (string), confidence_threshold (float)
3. WHEN configuration structure changes THEN the classes SHALL reflect the same structure
4. IF configuration validation is needed THEN it SHALL be simple type checking, not model instantiation

### Requirement 6: Model Factory Pattern

**User Story:** As a developer, I want a clean way to create models from configuration data, so that model instantiation is separate from configuration but still convenient.

#### Acceptance Criteria

1. WHEN I need to create a model THEN I SHALL use a factory function that takes configuration data
2. WHEN model creation fails THEN it SHALL not affect other parts of the system
3. WHEN different model types are needed THEN the factory SHALL handle them based on configuration
4. IF model files don't exist THEN the factory SHALL provide clear error messages

### Requirement 7: Breaking Changes for Model Access Patterns

**User Story:** As a developer, I want configuration classes to only provide data and never return loaded models, so that the heavy import issue cannot reoccur even if code expects loaded models.

#### Acceptance Criteria

1. WHEN accessing configuration properties THEN they SHALL never return loaded model instances
2. WHEN existing code expects loaded models from configuration THEN it SHALL be updated to use factory patterns instead
3. WHEN configuration classes are used THEN they SHALL only provide configuration data, never instantiated objects
4. IF existing code breaks due to these changes THEN clear migration documentation SHALL explain how to update the code