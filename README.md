# Chest X-Ray Multiclass Classification (Normal, Pneumonia, Tuberculosis, Unknown)

This project implements a deep learning pipeline to classify chest X-ray images into multiple categories: **Normal**, **Pneumonia**, **Tuberculosis**, and **Unknown**. It leverages a hybrid architecture combining **Convolutional Neural Networks (CNN)**, **Kolmogorov-Arnold Networks (KAN)**, and **ResNet** blocks to improve classification accuracy.

---

## 📁 Dataset

* **Source**: Kaggle dataset – [Combined Unknown Pneumonia and Tuberculosis](https://www.kaggle.com/datasets/rifatulmajumder23/combined-unknown-pneumonia-and-tuberculosis)
* **Structure**:

  ```
  data/
    ├── train/
    │   ├── Normal/
    │   ├── Pneumonia/
    │   ├── Tuberculosis/
    │   └── Unknown/
    ├── val/
    └── test/
  ```
* Images are RGB, resized to **150x150**, and loaded using TensorFlow’s `image_dataset_from_directory`.

---

## 📌 Objective

To build a robust multiclass classifier that can assist in early screening and detection of thoracic diseases by:

* Distinguishing between multiple conditions.
* Maximizing performance on unseen test data.

---

## 🧠 Model Architecture

### ✅ Hybrid CNN + KAN + ResNet

* **CNN**: Extracts spatial features from X-ray images.
* **KAN (Kolmogorov-Arnold Network)**: Models complex nonlinear relations between features.
* **ResNet Block**: Adds skip connections to deepen the network without vanishing gradients.

### 🔧 Architecture Summary:

* Input: `(150, 150, 3)`
* Multiple `Conv2D` + `MaxPooling` layers
* **Residual blocks** to enhance learning
* Flatten → Dense → Dropout
* Final output: `Dense(4, activation='softmax')`

---

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ChestXRay-Multiclass-Classification.git
   cd ChestXRay-Multiclass-Classification
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it under the `data/` directory.

---

## 🧪 Training and Evaluation

The model is trained using:

* **Adam** or **RMSProp** optimizer
* **Categorical crossentropy** loss
* `EarlyStopping` and `ModelCheckpoint` for performance

### ✅ Metrics:

* Accuracy
* Precision, Recall, F1-score
* Confusion matrix (plotted using Seaborn)
* Classification report from `sklearn`

---

## 📊 Results

| Class        | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| Normal       | …         | …      | …        |
| Pneumonia    | …         | …      | …        |
| Tuberculosis | …         | …      | …        |
| Unknown      | …         | …      | …        |

*(Metrics are filled after training on full dataset)*

---

## 📷 Sample Predictions

Visualizations of predictions with true vs predicted labels are generated during testing.

---

## 🚀 Future Work

* Improve model accuracy with data augmentation.
* Integrate Grad-CAM for interpretability.
* Deploy the model as a web app using Streamlit or Flask.

---

## 📚 References

* [KAN Paper](https://arxiv.org/abs/2306.17577)
* [ResNet](https://arxiv.org/abs/1512.03385)
* TensorFlow Docs

---

## 🙌 Acknowledgments

* [Rifatul Majumder](https://www.kaggle.com/rifatulmajumder23) for the dataset.
* TensorFlow and Keras community.

---

## 📄 License

This project is licensed under the MIT License.
