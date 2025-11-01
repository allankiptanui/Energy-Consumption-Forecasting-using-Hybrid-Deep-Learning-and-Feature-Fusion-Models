# ⚡️ Energy Consumption Forecasting using Hybrid Deep Learning and Feature Fusion Models

This repository implements **energy consumption forecasting** using multiple deep learning architectures — including GRU, LSTM, CNN-GRU-Attention, Hybrid, and ARIMA — with **automated preprocessing, training, evaluation, and visualization** workflows.

---

## 🧩 Overview

Accurate forecasting of household energy consumption is crucial for optimizing energy management, load balancing, and smart grid efficiency.  
This project compares several models and integrates **feature fusion** and **attention mechanisms** to enhance prediction performance.

### Implemented Models:
- 🌀 **GRU (Gated Recurrent Unit)**
- 🔁 **LSTM (Long Short-Term Memory)**
- 🧠 **CNN-GRU with Attention (Hybrid Sequence Model)**
- ⚙️ **Hybrid Deep Learning Model**
- 📈 **ARIMA (Classical Time Series Model)**

---

## 🧠 Features

✅ Automated preprocessing (cleaning, resampling, feature engineering)  
✅ Train/test splits and normalization  
✅ Multiple model architectures with unified training workflow  
✅ Evaluation with metrics (MAE, RMSE, MAPE, SMAPE)  
✅ Visualization:
- Forecast vs Actual
- Training loss curve
- Residual analysis
- Error histograms
- Attention heatmaps (for hybrid/attention models)

---

## 🛠️ Requirements

- Python **3.10+**
- PyTorch
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- statsmodels (for ARIMA)

All dependencies are listed in `requirements.txt`.

---

## ⚙️ Installation (macOS/Linux)

Open your terminal and run:

```bash
# 1️⃣ Clone the repository
git clone https://github.com/allankiptanui/Energy-Consumption-Forecasting-using-Hybrid-Deep-Learning-and-Feature-Fusion-Models.git
cd Energy-Consumption-Forecasting-using-Hybrid-Deep-Learning-and-Feature-Fusion-Models

# 2️⃣ Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

 
# 5️⃣ Run your desired model
python -m experiments.run_training_workflow --model gru
