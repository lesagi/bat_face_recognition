# CVAT Setup Guide for Bat Face Annotation

## Quick Setup (Docker - Easiest)

### 1. Install Docker
```bash
# On macOS with Homebrew
brew install docker
# Or download Docker Desktop from docker.com
```

### 2. Clone and Start CVAT
```bash
git clone https://github.com/opencv/cvat
cd cvat
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### 3. Access CVAT
- Open browser: `http://localhost:8080`
- Create admin account
- Login and start annotating

## Alternative: CVAT.ai (Cloud - Even Easier)
- Go to: https://app.cvat.ai
- Sign up for free account
- Start annotating immediately (no setup)

## Creating Bat Face Annotation Project

### 1. Create New Project
- Name: "Bat Face Landmarks"
- Labels to create:
  - `face` (rectangle)
  - `left_eye` (point)
  - `right_eye` (point)  
  - `nose` (point)

### 2. Upload Images
- Select your best 200-300 bat face images
- Upload to CVAT project

### 3. Annotation Process
1. **Draw face bounding box** around entire face
2. **Mark left eye** with point annotation
3. **Mark right eye** with point annotation
4. **Mark nose tip/center** with point annotation
5. **Save** and move to next image

### 4. Export Data
- Format: "YOLO 1.1"
- Download annotations
- You'll get the exact folder structure needed for training

## Annotation Tips for Bat Faces

### Consistency is Key:
- Always mark **same eye positions** (center of eye, not corner)
- Use **consistent nose point** (tip or center)
- Include **various poses** (front, slight angles)
- **Quality > Quantity** - better to have 200 perfect annotations than 500 sloppy ones

### What Makes Good Training Data:
- Clear face visibility
- Different lighting conditions  
- Various bat orientations
- Different individuals (genetic diversity)
- High image quality

## Troubleshooting

### If CVAT is slow:
- Use CVAT.ai cloud version
- Or try Label Studio (also free)

### If annotations look wrong:
- Double-check class names match exactly
- Ensure points are placed consistently
- Verify export format is "YOLO 1.1"

## Ready for Training?
Once you have 200+ annotated images:
1. Export from CVAT in YOLO format
2. Split into train/val/test (70/20/10)
3. Use the training code from the guide
4. Train for 100+ epochs
5. Integrate into your pipeline

**Estimated time:** 2-3 days for annotation, 4-6 hours for training 