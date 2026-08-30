# `face_pose` training data

**The data is not here.** It lives at
`legacy/face_annotation_eyes_nose/yolo_dataset_keypoints/`, alongside the
conversion and augmentation scripts that built it (`convert_cvat_to_yolo.py`,
`create_augmented_dataset.py`, `check_augmented_labels.py`) and the trained
`best.pt`. That tree is 873 M and was left in `legacy/` rather than moved,
because the scripts reference each other by relative path.

It is 43 labeled faces: eyes + nose keypoints on **tight square face crops that
fill the frame**.

## The constraint that breaks pose if you forget it

`models/preprocessing/face_pose.pt` was trained on those tight crops, so
**full-frame inference wrecks the keypoints** — border and nose misdetections
produce tilted alignment.

Segmentation runs on the full frame; pose runs on the **mask-centered square
crop**, with keypoints mapped back to frame coordinates afterwards. The working
implementation is `pose_on_mask_crop` in
`scripts/build_variants_from_frames.py`.

Related: crops are centered on the **mask**, not the eyes. Eyes are used only to
straighten. Eye-centered cropping was tried and gave worse recognition.
