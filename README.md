

# 🧠 Brain Tumor Classification

This project is a graduation project that focuses on developing a machine learning system to classify brain tumors from MRI images. The model is designed to assist medical professionals by providing a reliable, automated second opinion, potentially speeding up diagnosis and improving treatment planning.

---

## 📌 Objectives

* Provide an easy-to-use tool to support doctors in classifying different types of brain tumors based on MRI scans.
* Reduce the time required for manual image analysis and diagnosis.
* Offer a system that can help in early detection, improving the chances of successful treatment.
* Build a system that does **not** depend on complex CNN architectures like ResNet or Inception, but rather a carefully designed custom model that achieves high performance.
* Ensure the solution is accessible and can be extended for future enhancements.

---

## 🎯 Benefits

* **Time Efficiency**: The model automates tumor classification, saving valuable time for doctors.
* **Accuracy**: With a test accuracy of **99.8%**, the system delivers highly reliable results.
* **Assistive Technology**: Acts as a supporting tool, helping reduce human error.
* **Scalability**: Can be integrated into larger diagnostic systems or extended to include more tumor types and medical conditions.

---

## ⚙ Technical Perspective

Our brain tumor classification system is built using a **Convolutional Neural Network (CNN)** implemented in TensorFlow/Keras. The model is lightweight yet highly accurate, achieving **99.8% accuracy** on unseen test data.

### 📂 Dataset

* MRI brain scan images organized into separate directories for training and testing.
* Image resolution standardized at **170 × 170 × 3 (RGB)**.
* Loaded using `image_dataset_from_directory` with categorical labels and shuffling for training.

### 🏗 Model Architecture

The CNN has been designed to balance complexity and performance:

* **Input Layer**

  * `Rescaling(1./255)` — Normalize pixel values to \[0, 1].
* **Convolutional Blocks**

  * `Conv2D(64, (5,5))` + `MaxPooling2D(3,3)` + `Dropout(0.3)`
  * `Conv2D(64, (5,5))` + `MaxPooling2D(3,3)` + `Dropout(0.3)`
  * `Conv2D(128, (4,4))` + `MaxPooling2D(2,2)` + `Dropout(0.3)`
  * `Conv2D(128, (4,4))` + `MaxPooling2D(2,2)` + `Dropout(0.3)`
* **Dense Layers**

  * `Flatten()`
  * `Dense(512, activation="relu")` + `Dropout(0.5)`
  * `Dense(num_classes, activation="softmax")`

### ⚡ Optimization

* **Optimizer:** Adam with learning rate `0.001`, β₁ = 0.85, β₂ = 0.9925
* **Loss Function:** categorical cross-entropy
* **Metrics:** accuracy

### 🛠 Training Configuration

* **Callbacks:**

  * `ModelCheckpoint`: Save best model based on validation accuracy.
  * `ReduceLROnPlateau`: Adjust learning rate on validation loss plateaus.
  * `EarlyStopping`: Stop training when validation loss stops improving.
  * **Custom learning rate scheduler**: Dynamically reduces LR when val\_accuracy reaches thresholds (96%, 99%, 99.35%).
* **Epochs:** Up to 28 (with early stopping enabled)
* **Batch sizes:** 128 (train), 256 (test)

### 📊 Results

* **Test accuracy:** 99.8%
* **Confusion matrix:** Nearly perfect classification across all categories
* **Classification report:** High precision, recall, and F1-score across all tumor classes

### 🖼 Visualizations

* Confusion matrix plotted using seaborn for intuitive understanding of model performance.

### 🛠 Tools & Libraries

* TensorFlow/Keras — model creation and training
* scikit-learn — metrics and evaluation
* matplotlib/seaborn — plots and visualizations
* numpy — data manipulation

---

## 🚀 How to Run

```bash
# Ensure you have Python 3.x installed
# Install required libraries
pip install tensorflow matplotlib seaborn scikit-learn numpy

# Train the model (example)
python train.py  # or run your Jupyter notebook

# Evaluate the model
python evaluate.py  # or run the evaluation cells in notebook
```

---

## 📌 Future Enhancements

* Add more advanced data augmentation techniques to increase robustness.
* Integrate into a web or desktop app for easier use by doctors.
* Explore explainable AI techniques (e.g., Grad-CAM) to highlight tumor regions.

---

## 👨‍💻 Authors

* Karim mohamed abdelfatah
*  Mohamed Ahmed Diaa

* Elshorouk Academy — Computer Science Graduation Project


