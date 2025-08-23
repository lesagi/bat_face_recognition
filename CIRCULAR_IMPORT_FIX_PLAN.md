# Circular Import Issue - Implementation Plan

## Problem Summary
Circular import dependency causing mutex lock failures during module initialization:
- `input_processor.py` → `image_processor` → `config` → `model_config.py` → `ultralytics.YOLO`
- Heavy `ultralytics` import at module level creates deadlock in circular dependency chain

## Root Cause
1. **Module-level ultralytics import** in `config/model_config.py` line 9
2. **Circular dependency chain** through package `__init__.py` files
3. **Heavy dependency loading** during import resolution phase

## Solution Strategy

### Phase 1: Immediate Fix - Lazy Import Pattern
**Priority: HIGH - Fixes the immediate mutex lock issue**

1. **Fix `config/model_config.py`**
   - Move `from ultralytics import YOLO` inside methods/functions
   - Use lazy import pattern to defer heavy dependency loading
   - Keep type hints but import only when needed

2. **Simplify `image_processor/__init__.py`**
   - Remove complex imports that create circular dependencies
   - Use direct module imports instead of package-level imports

### Phase 2: Structural Improvements
**Priority: MEDIUM - Prevents future circular import issues**

3. **Refactor Import Dependencies**
   - Move heavy model imports to lazy loading patterns
   - Use dependency injection for model instances
   - Implement factory patterns for model creation

4. **Package Structure Optimization**
   - Minimize `__init__.py` complexity
   - Use explicit imports instead of package-level exports
   - Separate configuration from model instantiation

### Phase 3: Long-term Architecture
**Priority: LOW - Improves overall maintainability**

5. **Dependency Inversion**
   - Create abstract interfaces for model operations
   - Implement plugin-style architecture for different model types
   - Use configuration-driven model loading

## Implementation Steps

### Step 1: Fix model_config.py (CRITICAL)
```python
# Before (problematic):
from ultralytics import YOLO  # Heavy import at module level

# After (fixed):
def _import_yolo():
    from ultralytics import YOLO
    return YOLO
```

### Step 2: Simplify image_processor/__init__.py
```python
# Before (creates circular dependency):
from .transforms import ImageTransforms
from config import ImageModelConfig

# After (simplified):
from .processor import ImageProcessor
# Remove problematic imports, use direct imports where needed
```

### Step 3: Update import patterns in consuming modules
- Use direct imports: `from image_processor.transforms import ImageTransforms`
- Avoid package-level imports that trigger circular chains

## Expected Outcomes
1. **Immediate**: Mutex lock error resolved
2. **Short-term**: Faster import times due to lazy loading
3. **Long-term**: More maintainable and flexible architecture

## Risk Assessment
- **Low Risk**: Changes are isolated to import patterns
- **High Impact**: Fixes critical blocking issue
- **Backward Compatible**: No API changes required

## Testing Strategy
1. Test import resolution: `python -c "from image_processor import ImageTransforms"`
2. Test pipeline functionality: Run existing test scripts
3. Verify no regression in model loading performance
4. Test all entry points that use image processing

## Files to Modify
1. `app/config/model_config.py` - Lazy import ultralytics
2. `app/image_processor/__init__.py` - Simplify imports
3. `app/siamese_network/input_processor.py` - Update import pattern
4. Any other files importing from image_processor package level

## Success Criteria
- [ ] No mutex lock errors during import
- [ ] All existing functionality works unchanged
- [ ] Import time improved or unchanged
- [ ] No circular import warnings
- [ ] All tests pass