# 🧠 Neural Networks — AIML Module Project

> An end-to-end Neural Networks project spanning two domains: **Signal Quality Classification** using a Multi-Layer Perceptron on tabular sensor data, and **Street View Digit Recognition** using a CNN on the SVHN (Street View House Numbers) image dataset.

---

## 📁 Project Structure

```
signal-classification-svhn-digit-recognition/
│
├── NN_Project_Jesmine.ipynb          # Main Jupyter Notebook
├── NN_Project_Jesmine.html           # HTML export of the notebook
│
├── NN_Project_Data_-_Signal.csv      # Signal quality dataset  (1599 rows × 12 cols)
│
├── train_32x32.h5                    # SVHN train set  (h5py format)
├── test_32x32.h5                     # SVHN test set   (h5py format)
│
└── README.md
```

---

## 🗂️ Project Overview

| Part | Domain | Technique | Marks |
|------|--------|-----------|-------|
| Part A | Electronics & Telecom — Signal Quality | MLP Neural Network | 30 |
| Part B | Autonomous Vehicles — Street View Digit Recognition | CNN Neural Network | 30 |
| **Total** | | | **60** |

---

## 📡 Part A — Signal Quality Classification (30 Marks)

**Domain:** Electronics and Telecommunication  
**Context:** A communications equipment manufacturer wants to predict the quality/strength of signals emitted by their equipment, using measurable signal parameters collected during testing.  
**Objective:** Build and improve a Neural Network classifier that predicts `Signal_Strength` from 11 numeric parameters.

---

### Dataset: `NN_Project_Data_-_Signal.csv`

- **Rows:** 1,599 signal test records
- **Columns:** 12 (11 numeric parameters + 1 target)
- **Missing values:** 0 (clean dataset)
- **Duplicate records:** 240 → requires deduplication

#### Features (all `float64`)

| Column | Description |
|--------|-------------|
| `Parameter 1` | Measurable signal attribute 1 (range: 4.6 – 15.9) |
| `Parameter 2` | Measurable signal attribute 2 (range: 0.12 – 1.58) |
| `Parameter 3` | Measurable signal attribute 3 (range: 0.0 – 1.0) |
| `Parameter 4` | Measurable signal attribute 4 (range: 0.9 – 15.5) |
| `Parameter 5` | Measurable signal attribute 5 |
| `Parameter 6` | Measurable signal attribute 6 |
| `Parameter 7` | Measurable signal attribute 7 |
| `Parameter 8` | Measurable signal attribute 8 |
| `Parameter 9` | Measurable signal attribute 9 |
| `Parameter 10` | Measurable signal attribute 10 |
| `Parameter 11` | Measurable signal attribute 11 |

#### Target Variable: `Signal_Strength` (`int64`)

| Class | Count | Percentage |
|-------|-------|-----------|
| `5` | 681 | 42.6% |
| `6` | 638 | 39.9% |
| `7` | 199 | 12.4% |
| `4` | 53 | 3.3% |
| `8` | 18 | 1.1% |
| `3` | 10 | 0.6% |

> ⚠️ **Class Imbalance:** 6-class classification with significant imbalance — classes `3` and `8` together account for less than 2% of records. Labels must be **one-hot encoded** (via `to_categorical`) before feeding into the Neural Network.

> ⚠️ **Duplicates:** 240 duplicate rows present — must be handled before splitting and modelling.

---

### Steps & Tasks

**1. Data Import & Understanding**
- Read CSV → (1,599 × 12)
- Check null % per attribute (all 0%)
- Detect and handle 240 duplicate records
- Visualise `Signal_Strength` distribution (6-class bar/count plot)
- Share at least 2 insights from initial data analysis

**2. Data Preprocessing**
- X / Y split: `Parameter 1`–`Parameter 11` vs. `Signal_Strength`
- Train/test split: **70:30**
- Print shapes of `X_train`, `X_test`, `y_train`, `y_test` and verify sync
- Normalise features with `MinMaxScaler` or `StandardScaler`
- Transform labels using **one-hot encoding** (`to_categorical`) for Neural Network compatibility

**3. Model Training & Evaluation**
- **Model 1 (Base):** Design MLP architecture (Dense layers + activation functions + output layer with `softmax`)
- Train and plot:
  - Training Loss vs. Validation Loss (epochs)
  - Training Accuracy vs. Validation Accuracy (epochs)
- **Model 2 (Improved):** Redesign architecture (e.g., add layers, change neurons, add Dropout, tune learning rate)
- Plot same visuals for Model 2 and compare both models — share insights on overfitting/underfitting behaviour

---

## 🚗 Part B — Street View Digit Recognition / SVHN (30 Marks)

**Domain:** Autonomous Vehicles / Computer Vision  
**Context:** Recognising multi-digit numbers in street-level photographs is critical for modern map-making. Google Street View contains hundreds of millions of geo-located panoramic images where address numbers must be automatically transcribed to pinpoint building locations accurately. This project uses the **SVHN (Street View House Numbers)** dataset — real-world images of house numbers, harder than MNIST due to distractors, varying fonts, lighting, and occlusions.  
**Objective:** Build a CNN digit classifier on single-digit SVHN images.

---

### Dataset: SVHN — `train_32x32.h5` + `test_32x32.h5`

- **Format:** HDF5 (`.h5`) files — read with `h5py`
- **Image size:** 32 × 32 pixels, RGB (3 channels)
- **Classes:** 10 digits (0–9)
- **Source:** Google Street View house number images

| Split | Description |
|-------|-------------|
| `X_train` | Training images — shape `(32, 32, 3, N_train)` |
| `y_train` | Training labels (digit 0–9) |
| `X_test` | Test images — shape `(32, 32, 3, N_test)` |
| `y_test` | Test labels (digit 0–9) |

> ⚠️ **Reshape required:** SVHN h5 files store images in `(H, W, C, N)` format — must be transposed to `(N, H, W, C)` before feeding into Keras/TensorFlow.

> ⚠️ **Pixel normalisation:** Raw pixel values range 0–255 → divide by 255 to normalise to [0, 1].

> ⚠️ **Label encoding:** Labels must be one-hot encoded (`to_categorical`) for multi-class Neural Network output.

---

### Steps & Tasks

**1. Data Import & Exploration**
- Read `.h5` files using `h5py` into variables
- Print all keys present in the `.h5` file
- Assign `X_train`, `X_test`, `y_train`, `y_test` from file keys

**2. Data Visualisation & Preprocessing**
- Print shapes of all 4 splits and verify X/Y alignment
- Visualise the first 10 training images with their labels
- Reshape images from `(H, W, C, N)` → `(N, H, W, C)`
- Normalise pixel values: divide by 255.0
- One-hot encode labels with `to_categorical`
- Print total number of classes in the dataset (10)

**3. Model Training & Evaluation**
- **Design CNN architecture:**
  ```
  Input (32×32×3)
  → Conv2D + ReLU → MaxPooling
  → Conv2D + ReLU → MaxPooling
  → Flatten
  → Dense (ReLU)
  → Dropout
  → Dense (Softmax, 10 classes)
  ```
- Train with best-fit parameters (`batch_size`, `epochs`, `optimizer`)
- Evaluate with Accuracy and Loss metrics
- Plot:
  - Training Loss vs. Validation Loss over epochs
  - Training Accuracy vs. Validation Accuracy over epochs
- Share observations on convergence, overfitting/underfitting

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Notebook:** Jupyter Notebook
- **Libraries:**

```
pandas             # Tabular data handling (Part A)
numpy              # Array operations
matplotlib         # Loss/accuracy plots, image visualisation
seaborn            # Distribution plots
scikit-learn       # Train-test split, normalisation
tensorflow / keras # Neural Network design, training, evaluation
h5py               # Reading SVHN .h5 dataset files (Part B)
```

---

## ⚙️ Setup & Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/neural-networks-aiml-project.git
cd neural-networks-aiml-project

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow h5py jupyter

# Launch the notebook
jupyter notebook NN_Project_Jesmine.ipynb
```

---

## 📈 Key Findings (Summary)

**Part A — Signal Quality Classification:**
- Dataset: **(1,599 × 12)** — completely clean with zero nulls, but **240 duplicate rows** must be removed first
- 6-class target (`Signal_Strength` 3–8) with heavy imbalance: classes `5` and `6` together make up 82.5% of records, while classes `3` and `8` are extremely rare (<2% combined)
- Labels must be one-hot encoded before training the Neural Network
- Model 2 (improved architecture) is expected to show better generalisation by addressing overfitting through Dropout or architecture adjustments

**Part B — SVHN Digit Recognition:**
- SVHN is a significantly harder problem than MNIST due to real-world variability: lighting, shadows, distractor digits, varying fonts and orientations
- HDF5 format requires shape transposition before use — a common preprocessing pitfall
- CNN architectures with Conv → Pool → Dense layers are well-suited for this 32×32 RGB classification task
- Training and validation loss/accuracy curves are critical diagnostics for detecting overfitting early

---

## 📚 References

- SVHN Dataset: Yuval Netzer et al., *Reading Digits in Natural Images with Unsupervised Feature Learning*, NIPS 2011 — [http://ufldl.stanford.edu/housenumbers](http://ufldl.stanford.edu/housenumbers)
- h5py documentation: [https://docs.h5py.org/en/stable/](https://docs.h5py.org/en/stable/)

---

## 📋 Submission Checklist

- [x] `.ipynb` notebook with all code, outputs, and markdown explanations
- [x] `.html` export of the notebook
- [x] All code cells have visible outputs
- [x] Insights documented after every analysis step
- [x] No plagiarism

---

## 📄 License

This project was completed as part of the **Great Learning AIML Programme** coursework.  
Educational use only.

---

*Made with 🧠 and Python*
