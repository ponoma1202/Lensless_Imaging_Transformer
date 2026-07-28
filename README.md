# Pan Transformer Adaptation for ConvRML Experiments

This repository is an adaptation of the original **Lensless Imaging Transformer** (LIT) implementation from Pan et al., *“Image reconstruction with transformer for mask-based lensless imaging,”* Optics Letters, 2022. The original model was used as a baseline in the ConvRML paper.  

This fork documents the modifications we made to support our ConvRML-style training, evaluation, and dataset preprocessing pipeline.

## Summary of Changes

### `train.py`

- Added Weights & Biases (wandb) logging for training metrics and image visualizations.
- Added intermediate checkpoint dictionary saving every 5,000 training steps.
- Updated metric tracking to match the main ConvRML code for more consistent comparison.

### `modules.py`

- Added support for rectangular kernel sizes and images.
- Updated the encoder to compute feature-map dimensions explicitly after each patch embedding layer.
- Removed the assumption that input dimensions are square or exactly divisible by 4, 8, and 16.

### `inference.py`

- Added a custom inference script to match the inference procedure used for ConvRML.

### `utils/data_prepare.py`

- Adapted the ConvRML preprocessing code for the Parallel Lensless Dataset to the Lensless Imaging Transformer

### `utils/data_utils.py`

- Added a `get_test_loader` method for inference.
- Extended the original data-loading structure, which only included training and validation dataloaders, to also support a test split.

### `datasets/prepare_dataset.py`

- Updated dataset splitting to match the ConvRML train/validation/test split structure.