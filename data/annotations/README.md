# `data/annotations/` — training data for the preprocessing models

The YOLO models in `models/preprocessing/` are trained from here. Before the
2026-08-30 reorganisation this data was scattered across six locations under
`data/interim/` and `data/processed/`, filed as "intermediate" — which it isn't:
it is hand-annotated ground truth, and it is the only copy.

Grouped by **which model it trains**, because that is the question you are
actually asking when you come back to it.

## `face_seg/` → `models/preprocessing/face_seg.pt`

Ultralytics segmentation format: one `.txt` per image, class `0`, normalised
polygon vertices.

| set | imgs | source | note |
|---|---|---|---|
| `mauritius/general/` | 99 | full-frame stills, 2252×4000 | was `interim/mauritius/segmentation/general` |
| `mauritius/frontal_only/` | 49 | face crops, already split | has `data.yaml`, `images/{train,val}`, `_label_check.png` |
| `rousettus/still/` | 27 | stills, 1800×4000 | was `interim/rousettus/still/segmentation` |
| `rousettus/still_alt/` | 26 | stills, 1800×4000 | **a different annotation pass** — see below |
| `rousettus/video/` | 60 | video frames | plus the CVAT export `annotations.xml` |
| `rousettus/video_yolo/` | 60 | the same 60, split | `images|labels/{train,val}` + `data.yaml` |

### `rousettus/still_alt` is not a duplicate

It was `processed/siamese_input_aligned_picsum/rous_images+labels`. Only **5 of
its 26 stems** overlap with `rousettus/still/` (27), and the two use different
filename conventions — `4--IMG_20250518_150105.jpg` vs
`babyis_IMG_20250518_151826.jpg`. Both are genuine annotation passes; neither
supersedes the other. Kept separate rather than merged because the overlapping 5
may carry different polygons and nobody has checked.

### `mauritius/general` absorbed a near-duplicate

`processed/siamese_input_aligned_picsum/mauritius_images+labels` held the same
99 images and 99 labels. **197 of those 198 files were byte-identical** to
`interim/mauritius/segmentation/general`, so the picsum copy was deleted and the
interim copy kept.

The single exception was `20230723_193209.jpg` — same 2252×4000 dimensions,
different bytes (5.7 M vs the 3.9 M kept here), i.e. re-encoded, not re-shot.
All 99 labels matched exactly, so the annotation was unambiguous either way and
the re-encoded variant was deleted too. This directory is now exactly 99 images
(89 `.jpg` + 10 `.png`) and 99 labels, one-to-one with no orphans on either side.

## `face_pose/` → `models/preprocessing/face_pose.pt`

The keypoint data lives outside `data/`, at
`legacy/face_annotation_eyes_nose/yolo_dataset_keypoints/`. See
[`face_pose/README.md`](face_pose/README.md) — and note the mask-crop
constraint recorded there before retraining anything.
