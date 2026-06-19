# EEG-Based Neurological Disease Classification

Deep learning exploration for 3-class classification of neurological conditions — **Alzheimer's Disease**, **Frontotemporal Dementia (FTD)**, and **Healthy** controls — using ICA-decomposed EEG signals. Three temporal architectures are implemented and compared: LSTM, GRU, and DNN.

This repository is part of a broader study on integrating spatial (MRI) and temporal (EEG) insights for Alzheimer's disease detection. The MRI-based component (InceptionV3 + ResNet50) is in a [separate repository](https://github.com/HarishDeepak/Alzheimer-Detection-using-ResNet50-and-InceptionV3).

## Dataset

- **Source**: ICA-decomposed EEG data (`output2.csv`)
- **Features**: 19 independent components (columns `s1`–`s19`) — output of Independent Component Analysis applied to raw multi-channel EEG recordings
- **Target classes**: Alzheimer (1), Frontotemporal Dementia (2), Healthy (3)
- **Size**: ~117 total samples; split 70% train / 30% test (random_state=111), with an additional 30% of train held as validation

| Split | Samples |
|---|---|
| Train | ~57 |
| Validation | ~25 |
| Test | 35 |

## Models

### LSTM
```
Input: (19, 1)
  → LSTM(500, return_sequences=True)
  → Flatten
  → Dense(3, softmax)
Total params: ~1.0M
```
- Optimizer: Adam (lr=0.001, exponential LR decay)
- Loss: CategoricalCrossentropy
- Callbacks: EarlyStopping (patience=10), ModelCheckpoint

### GRU
```
Input: (19, 1)
  → GRU(500, return_sequences=True)
  → Flatten
  → Dense(3, softmax)
Total params: ~0.78M
```
Same training protocol as LSTM.

### DNN (Deep Neural Network)
```
Input: (19,)
  → Dense(2548) → BatchNorm → Dropout(0.25)
  → Dense(3822) → BatchNorm → Dropout(0.27)
  → Dense(5096) → BatchNorm → Dropout(0.30)    ← bottleneck
  → Dense(3822) → BatchNorm → Dropout(0.27)
  → Dense(2548) → BatchNorm → Dropout(0.25)
  → Dense(3, softmax)
Total params: ~58.5M
```
EarlyStopping with patience=10; best checkpoint saved at epoch 4.

## Training Setup

- **Optimizer**: Adam with exponential learning rate decay (`lr = 0.001 × e^(-epoch/10)`)
- **Loss**: CategoricalCrossentropy (one-hot labels)
- **Callbacks**: EarlyStopping, ModelCheckpoint (best val_accuracy saved), LearningRateScheduler
- **Batch size**: 32
- **Max epochs**: 100 (LSTM), 50 (GRU), 50 (DNN)

## Setup

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

Notebooks require `output2.csv` — ICA-decomposed EEG data (not included due to size; available via project data pipeline).

1. Open `alzheimer_EEG_LSTM.ipynb` for LSTM and GRU models
2. Open `DNN.ipynb` for the deep feedforward network
3. Mount Google Drive if running in Colab and update the CSV path

## Project Context

This project investigates whether EEG temporal features alone can distinguish between three neurological conditions. ICA decomposition reduces multichannel EEG to independent source signals, removing artefacts (eye blinks, muscle noise) before classification. The three model families (RNN-based LSTM/GRU vs. dense DNN) were compared to evaluate whether sequential modelling of the 19 components provides an advantage over treating them as a flat feature vector.

## References

- Alzheimer's Association — [Frontotemporal Dementia](https://www.alz.org/alzheimers-dementia/what-is-dementia/types-of-dementia/frontotemporal-dementia)
- Related MRI project: [Alzheimer Detection using ResNet50 and InceptionV3](https://github.com/HarishDeepak/Alzheimer-Detection-using-ResNet50-and-InceptionV3)
