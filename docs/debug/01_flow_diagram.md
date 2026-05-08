# Siamese Network Training Pipeline -- Flow Diagram

> Generated: 2026-04-10  
> Source: `app/siamese_training/trainer.py`, `app/siamese_core/network.py`, `app/siamese_data/data_splitter.py`

---

## 1. Top-Level Training Flow

This diagram covers the entire lifecycle from CLI invocation through model save.

```mermaid
flowchart TD
    subgraph entry [CLI Entry Point]
        A[train_siamese.py::main] --> B[Parse CLI args:<br/>--bat-type, --data-source,<br/>--augmented-data, --background,<br/>--permute-labels]
    end

    B --> C[SiameseNetworkTrainer.__init__]

    subgraph init [Trainer Initialization]
        C --> D[load_config<br/>config.yml]
        D --> E[Mixed precision setup<br/>if advanced.mixed_precision=true]
        E --> F[Resolve input_dir<br/>from input_paths.bat_key by --background]
        F --> G[Set hyperparameters:<br/>epochs, batch_size from config]
        G --> H[Build Siamese model<br/>SiameseNetwork L1Dist .model]
        H --> I[Create optimizer + loss:<br/>Adam 1e-4, BCE reduction=none]
        I --> J[SiameseNetworkTrainingDataSplitter<br/>pair generation + preprocessing]
        J --> K{class_balancing<br/>enabled?}
        K -->|Yes| L[compute_global_class_distribution]
        K -->|No| M[Skip weighting]
        L --> N[ClassWeightCalculator<br/>INS / ISNS / ENS scheme]
        M --> N
        N --> O[Create train_batches / test_batches<br/>.batch .prefetch AUTOTUNE]
        O --> P[tf.train.Checkpoint setup]
    end

    P --> Q[trainer.fit]

    subgraph fit [fit method]
        Q --> R[_start_parent_run<br/>MLflow + output dir + stdout capture]
        R --> S[trainer.train loop]
        S --> T[trainer.test final eval]
        T --> U[trainer.save_model final]
        U --> V[_end_parent_run<br/>close MLflow + log file]
    end
```

---

## 2. Data Loading and Pair Generation Flow

This covers `SiameseNetworkTrainingDataSplitter.__init__` in detail.

```mermaid
flowchart TD
    A[Input: images_dirs_paths_list] --> B[Walk directories<br/>get_files_from_dir]
    B --> C[group_files_by_class<br/>filename_parser.py<br/>pattern: type--class--id]
    C --> D[__split_individual_images<br/>per-class train/test split<br/>by image ID not file]

    D --> E[__create_training_pairs]
    D --> F[__create_testing_pairs]

    subgraph trainPairs [Training Pair Generation]
        E --> E1[For each class:<br/>__create_anchor_pairs<br/>positive pairs label=1.0]
        E --> E2[For each class pair i,j:<br/>__create_negative_pairs<br/>negative pairs label=0.0]
        E1 --> E3[Combinations or<br/>Permutations of images]
        E2 --> E4[Cartesian product<br/>of images across classes]
        E3 --> E5[PairClassInfo.create_single_class]
        E4 --> E6[PairClassInfo.create_dual_class]
    end

    subgraph testPairs [Testing Pair Generation]
        F --> F1[Same logic as training<br/>but from test_class_files]
    end

    E5 --> G[Concatenate anchors + negatives<br/>into train_data Dataset]
    E6 --> G
    F1 --> H[Concatenate anchors + negatives<br/>into test_data Dataset]

    G --> I{permute_labels?}
    I -->|Yes| J[Shuffle labels across<br/>all pairs randomly<br/>destroys signal for null test]
    I -->|No| K[Keep original labels]

    J --> L[.map preprocess_twin_input_function<br/>per-image: read, decode, resize, /255]
    K --> L

    L --> M[Shuffle final datasets<br/>buffer_size = dataset size]
```

---

## 3. Training Loop Flow (per epoch)

```mermaid
flowchart TD
    A[Epoch start] --> B[Reset Recall + Precision metrics]

    B --> C[For each batch in train_batches]

    subgraph batchLoop [Batch Training Step]
        C --> D[train_step batch]
        D --> D1[Extract x=img1 img2, y=label,<br/>class_info from batch]
        D1 --> D2[Forward pass:<br/>yhat = siamese_model x, training=True]
        D2 --> D3[Per-sample loss:<br/>BCE y, yhat reduction=none]
        D3 --> D4{Weighting<br/>enabled?}
        D4 -->|Yes| D5[Compute anchor/negative weights TF<br/>+ per-class weights via lookup table]
        D4 -->|No| D6[loss = reduce_mean per_sample_loss]
        D5 --> D7[Normalize weights mean=1.0<br/>weighted_loss = per_sample * weights<br/>loss = reduce_mean weighted_loss]
        D6 --> D8[Gradient tape<br/>compute gradients]
        D7 --> D8
        D8 --> D9{Mixed<br/>precision?}
        D9 -->|Yes| D10[get_scaled_loss -> gradient<br/>-> get_unscaled_gradients]
        D9 -->|No| D11[tape.gradient directly]
        D10 --> D12[optimizer.apply_gradients]
        D11 --> D12
    end

    D12 --> E[Update Recall, Precision<br/>with batch y, yhat]
    E --> C

    C -->|All batches done| F["train_loss = last batch loss (BUG)<br/>train_recall = r.result<br/>train_precision = p.result<br/>train_f1 = 2*P*R / P+R"]

    F --> G[Run test evaluation]
    G --> H[_update_metric_history]
    H --> I[_log_epoch_metrics to MLflow]
    I --> J[_plot_and_log_artifacts]

    J --> K{test_loss < best_loss?}
    K -->|Yes| L[Save best_model_loss<br/>Update best_loss_value]
    K -->|No| M{test_f1 > best_f1?}
    L --> M
    M -->|Yes| N[Save best_model_f1<br/>Update best_f1_value]
    M -->|No| O[_check_early_stopping]
    N --> O

    O --> P{Early stop<br/>triggered?}
    P -->|Yes| Q[Restore best weights if configured<br/>Break training loop]
    P -->|No| R{epoch % 10 == 0?}
    R -->|Yes| S[Save periodic model + checkpoint]
    R -->|No| T[Next epoch]
    S --> T
```

---

## 4. Test Evaluation Flow

```mermaid
flowchart TD
    A[test method called] --> B[Reset Recall + Precision<br/>total_loss = 0, num_batches = 0]

    B --> C[For each batch in test_batches]

    subgraph testBatch [Test Batch]
        C --> D[Unpack: test_input, test_val,<br/>y_true, class_info]
        D --> E["yhat = siamese_model.predict(<br/>[test_input, test_val])"]
        E --> F[Update Recall, Precision<br/>with y_true, yhat]
        F --> G[batch_loss = unweighted BCE<br/>y_true, yhat]
        G --> H[total_loss += batch_loss<br/>num_batches += 1]
    end

    H --> C
    C -->|Done| I[avg_loss = total_loss / num_batches]
    I --> J[Return: avg_loss, recall, precision, f1]
```

---

## 5. Image Preprocessing Flow (per image inside tf.data.map)

```mermaid
flowchart TD
    A[file_path string tensor] --> B[tf.io.read_file]
    B --> C["try: tf.io.decode_png<br/>except: tf.io.decode_jpeg<br/>(BUG: try/except in graph mode)"]
    C --> D["load_config() <-- called per image (BUG)"]
    D --> E["tf.image.resize(img, input_edge, input_edge)"]
    E --> F[img / 255.0 normalize]
    F --> G[convert_image_dtype float32]
    G --> H{Channels?}
    H -->|1 grayscale| I[tf.repeat to 3 channels]
    H -->|4 RGBA| J[Slice to RGB :3]
    H -->|3 RGB| K[Pass through]
    I --> L[Return preprocessed tensor]
    J --> L
    K --> L
```

---

## 6. Post-Training Automation Flow

```mermaid
flowchart TD
    A[_run_post_training_automation] --> B[Load best_model_f1 path]
    B --> C{Model exists?}
    C -->|No| D[Skip automation]
    C -->|Yes| E[generate_predictions_from_config<br/>using original background images]
    E --> F[Save CSV + confusion matrix]
    F --> G[Load model for saliency]
    G --> H[SiameseModelSaliencyMapCreator<br/>integrated_gradients method]
    H --> I[Generate per-bat saliency images]
    I --> J[Save to model_output_dir/saliency/]
```

---

## 7. MLflow Lifecycle Flow

```mermaid
flowchart TD
    A[_start_parent_run] --> B[mlflow.set_tracking_uri<br/>mlflow.set_experiment]
    B --> C[mlflow.start_run parent]
    C --> D[_create_output_directory<br/>date_experimentId_runId]
    D --> E[Set tags: bat_type, species,<br/>data_source, augmented]
    E --> F[Log config snapshot artifact]
    F --> G[_log_hyperparameters<br/>all params to MLflow]
    G --> H[_log_class_balancing_to_mlflow]
    H --> I[_log_sample_images 5 random]

    I --> J[Training loop]
    J --> K[Per-epoch: _log_epoch_metrics<br/>parent run + nested epoch run]
    K --> L[Per-epoch: _plot_and_log_artifacts<br/>loss/recall/precision/f1 PNGs]

    L --> M[_end_parent_run]
    M --> N[Log training.log artifact]
    N --> O[mlflow.end_run]
```
