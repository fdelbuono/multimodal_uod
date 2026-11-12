# Real-Time Multi-Modal Underwater Object Detection

This repository contains the implementation developed for the MSc thesis *Real-Time Multi-Modal Underwater Object Detection*. It includes the camera–sonar calibration pipeline, the multi-modal detection and fusion pipeline, hybrid dataset generation, and the trained uni-modal models.

## Repository Structure

### `calibration/`

Contains the camera–sonar calibration implementation and evaluation:

- **`calibration_config.py`**: Loads the calibration parameters from a JSON calibration file.
- **Calibration JSON files**: Store the calibration parameters obtained from the different calibration experiments.
- **`calibration_pipeline.py`**: Implements the targetless camera–sonar calibration pipeline.
- **`sonar_camera_sensor_models.py`**: Contains the camera and sonar sensor models and their associated functions.
- **`testing_calibration.py`**: Implements the calibration testing and evaluation procedures.

### `fusion/`

Contains the camera–sonar fusion implementation and hybrid dataset tools:

- **`camera_sonar_fusion_pipeline.py`**: Implements the camera–sonar detection fusion pipeline and its evaluation.
- **`hybrids_datasets.py`**: Generates hybrid camera and sonar datasets using the annotations from both modalities.
- **`plot_hybrid_datasets.py`**: Visualizes examples of the generated hybrid datasets for inspection.

### `unimodal/`

Contains the uni-modal detection models:

- **`trained_models/`**: Contains the trained camera and sonar detection weights.
- **`yolov12-wiseiou/`**: YOLOv12 implementation with WIoU modifications.
- **`yolov12-coordconv/`**: YOLOv12 implementation with CoordConv modifications.

The YOLOv12 implementations are based on [Ultralytics](https://github.com/ultralytics/ultralytics). Full credit goes to Ultralytics for the original framework; the modifications were introduced as part of this thesis.

## Installation and Requirements

The project uses a Conda environment. To recreate the environment, clone the repository and run:

```bash
conda env create -f environment.yml
```

## Acknowledgements

Credit to [Ultralytics](https://github.com/ultralytics/ultralytics) for the original YOLO implementation and framework.