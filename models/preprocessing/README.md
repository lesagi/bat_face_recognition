# Preprocessing Model Weights

Active YOLO weights consumed by `bat_preprocessing` (video → still-image extraction, face crop, alignment).

| File | Source (pre-relocation) | Purpose |
|---|---|---|
| `face_seg.pt` | `app/background_replacement/model/best.pt` | YOLOv8 segmentation — bat face region |
| `face_pose.pt` | `legacy/face_annotation_eyes_nose/best.pt` | YOLOv8 pose — eyes + nose landmarks |

The Hydra preprocessing config (`configs/preprocessing/yolo.yaml`) points at these paths.

> Note: `face_seg.pt` was relocated from `app/background_replacement/model/`. The originally-canonical
> path in `app/config/config.yml` (`legacy/rousesttus_segmentation/.../bat_face_seg/weights/best.pt`)
> was not present in this checkout. If a different segmentation checkpoint is preferred, replace
> `face_seg.pt` in place — no code changes needed, the path stays the same.
