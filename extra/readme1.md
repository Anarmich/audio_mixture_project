# Deep U-Net Acoustic Separation Network (DU-ASNet)

## Overview

The Deep U-Net Acoustic Separation Network (DU-ASNet) is a neural network architecture designed for polyphonic sound event detection and localization. This framework provides tools for preparing data, augmenting datasets, training the model, and evaluating performance.

## Workflow

The following steps outline the process of preparing data, augmenting datasets, training the model, and evaluating performance.

### Step 1: Training Data Acquisition

Obtain the training data, which consists of audio recordings from MEC Vehicle B. Ensure that you have the audio files organized in the following directory structure:

data/
    train/
        road/
        wind/
        powertrain/
    valid/
    test/


### Step 2: Reorganize Files

Use the `reorganize_files.py` script to ensure that the audio files are organized correctly for the model to train:

```bash
python scripts/reorganize_files.py
```

Step 3: Pick the Split of Training, Testing, and Validation
Use the split.py script to split the dataset into training, validation, and testing sets:

```bash
python scripts/split.py
```

Step 4: Reset the Training Set
Insert augmentation to the training data using the augment.py script to generate augmented data:

```bash
python scripts/augment.py
```

Step 5: Introduce Mixtures
Generate audio mixtures using the equivalent files based on the previous step's selection. Use the augment_createmixtures.py script:

```bash
python scripts/augment_createmixtures.py
```

Step 6: Data Preparation
Prepare the data for training using the dataprep.py script:

```bash
python scripts/dataprep.py
```

Step 7: Training the Model
Use the train_normalize.py script to train the model. This script includes normalization and adaptive loss function mechanisms. If a validation set is used, it will also be included in the training loop:

```bash
python scripts/train_normalize.py
```

Step 8: Model Testing and Execution
After training the model, use the try_it.py script to test the model on new data:

```bash
python scripts/try_it.py
```

Detailed Instructions
1. Data Acquisition
Ensure you have the necessary audio files and store them in the specified directory structure.

2. File Reorganization
The reorganize_files.py script will move and rename the audio files to match the required format. This script ensures that all files are correctly labeled and placed in the appropriate directories.

3. Data Splitting
The split.py script will split your dataset into training, validation, and testing sets. This step is crucial for model evaluation and preventing overfitting.

4. Data Augmentation
The augment.py script generates augmented data to increase the diversity of your training set. This step improves the model's robustness and generalization capabilities.

5. Mixture Generation
The augment_createmixtures.py script creates audio mixtures from the augmented data. These mixtures are used as input for the neural network to learn how to separate the sources.

6. Data Preparation
The dataprep.py script processes the audio files and prepares them for training. This script handles normalization and other preprocessing steps.

7. Model Training
The train_normalize.py script trains the deep U-Net model using the prepared dataset. It includes mechanisms for monitoring training progress using TensorBoard and implements an adaptive loss function to balance the contributions of different audio sources.

8. Model Testing and Execution
The try_it.py script tests the trained model on new audio data. It allows you to evaluate the model's performance and visualize the results.


Contact
For further questions or issues, please contact 
