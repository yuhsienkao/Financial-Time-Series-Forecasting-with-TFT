# Stock Return Ranking Prediction with the Temporal Fusion Transformer

**Project Type:** Academic Research Project as a **Quantitative Research Assistant** under the supervision of **Prof. Hsiao, Chan-Tung**, Department of Finance, National Taiwan University.

**Status:** Successfully maintained, executed, and analyzed the results of the research pipeline.

---

## 1. Objective & Motivation

This research project aims to leverage state-of-the-art deep learning techniques to predict future stock return rankings based on historical financial data. The core of this project is the application of the **Temporal Fusion Transformer (TFT)**, a novel attention-based architecture designed specifically for multi-horizon time-series forecasting.

The motivation is to explore the predictive power of complex, non-linear models in identifying patterns within financial markets that traditional econometric models might miss. The TFT model was chosen due to its unique ability to incorporate static, known, and observed inputs, making it particularly well-suited for financial data.

---

## 2. Methodology & My Contribution

As a research assistant, my primary responsibility was to **maintain, debug, and operationalize the complete research pipeline** built with `PyTorch Lightning` and `pytorch-forecasting`. This involved a deep, hands-on understanding of the entire workflow, from data ingestion to model evaluation.

### Key Contributions:

* **Data Pipeline Management:**
    * Managed the data loading and cleaning process for the financial time-series dataset (`TFT_prep.csv`) using `pandas`.
    * Implemented robust data handling for missing values and ensured the integrity of the time series index across thousands of unique stocks (`StockID`).
    * Configured the `TimeSeriesDataSet` object, correctly defining static and time-varying variables, encoders, and target normalizers for the TFT model.

* **Model Training & Execution:**
    * Managed the high-performance model training process using `PyTorch Lightning`, configuring the `Trainer` with GPU acceleration (`accelerator="gpu"`), early stopping callbacks, and learning rate monitoring.
    * Successfully executed the training pipeline, ensuring the model converged and produced valid checkpoints for further analysis.

* **Results Analysis & Interpretation:**
    * Processed the raw output of the model to calculate prediction probabilities using `torch.nn.functional.softmax`.
    * Conducted post-prediction analysis, including calculating overall accuracy, generating confusion matrices, and visualizing the results with `matplotlib` and `seaborn` to interpret model performance and identify areas for future improvement.

***Note: The provided Jupyter Notebook (`old_tft (1).ipynb`) demonstrates the core logic and workflow of the research pipeline I was responsible for maintaining and executing.***

---

## 3. Key Technologies & Fields

* **Languages & Libraries:** Python (PyTorch Lightning, pytorch-forecasting, Pandas, NumPy)
* **Core Fields:** Quantitative Finance, Deep Learning, Time-Series Analysis, Machine Learning
* **Models:** Temporal Fusion Transformer (TFT)
