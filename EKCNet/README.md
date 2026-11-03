🚀 Training and Evaluation Guide

This guide provides step-by-step instructions for training the model with complete modalities, transferring it to missing-modality datasets, and running inference.
1. Pre-training on BraTS2020 (Complete Modalities)

Before transfer learning, the model needs to be pre-trained on a complete-modality dataset (e.g., BraTS2020).

Modify train.py:

    Update --path_image to point to your local BraTS2020 dataset directory.

    Set --freeze_encoder 'none' to train the entire network.

    Line 190: Set the CSV file to frame = pd.read_csv("BraTS2020_Training_5folds.csv").

    Line 190: Set fold = 2.

    Lines 351 & 372: Ensure no modalities are zeroed out for complete-modality training:
    python

input[:, [], ...] = 0.0  # Keep all modalities

    📥 Pre-trained Weights:
    The pre-trained model weights on BraTS2020 (complete modalities) are available at:
    quan_2_TC_0.8946437642380998_ED_0.8318433842143497_ET_0.8162086235510336_WT_0.9094836148055824.pt

2. Transfer Learning on Missing-Modality Datasets

After pre-training, you can transfer the learned knowledge to downstream datasets with missing modalities.

Modify train.py:

    Update --path_image to point to your target dataset.

    Set --freeze_encoder based on the missing modality scenario:

        Choose 'encoder1' or 'encoder2' depending on which branch's modalities are missing.

    Line 190: Update the CSV file:

frame = pd.read_csv("2021dataset_125_126.csv") 

    Lines 351 & 372: Adjust the zeroed modalities (input[:, [], ...] = 0.0) according to your specific missing-modality setting.

The training instructions and CSV files for the BraTS2023-PED dataset will be released soon.

3. Inference and Visualization

    inference.py: The script used to run predictions on the test set.

    visual.py: The script to generate and visualize the segmentation results 