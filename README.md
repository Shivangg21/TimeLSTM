
**TimeLSTM**

An interactive **Streamlit** application for **multi-step time series forecasting** using **LSTM (Long Short-Term Memory) networks**.
Designed for both data science professionals and non-technical users, this project makes deep learning–powered forecasting **accessible, customizable, and intuitive**.

---

**✨ Features**

* 📂 **Data Handling**

  * Upload CSV files with auto date-detection & mixed datatype support.

* 🧹 **Preprocessing**

  * Handles missing values, categorical encoding, and feature scaling.

* 📊 **Exploration & Visualization**

  * Interactive time series plots, histograms, correlation heatmaps, and seasonal decomposition.

* 🧠 **LSTM Model**

  * Customizable architecture (layers, neurons, forecast horizon) with GPU acceleration.

* ⚡ **Training Framework**

  * Adjustable epochs, batch size, and learning rate with real-time monitoring.

* 📈 **Results Analysis**

  * MSE, RMSE, MAE metrics, residual error inspection, and forecast visualization.

* 💾 **Export & Deployment**

  * Save results as CSV/plots, persistent model storage for reuse.

---

**⚙️ How It Works**

1. **Streamlit UI** – Provides an interactive web-based interface for model training and forecasting.

2. **Data Pipeline** – Upload, preprocess, and visualize datasets before training.

3. **LSTM Training** – Uses PyTorch to train customizable models with GPU support.

4. **Evaluation** – Generates forecast plots, error metrics, and residual analysis.

5. **Export Options** – Save trained models, forecasts, and plots for future use.

---

**🌍 Use Cases**

* 💹 **Finance** – Stock price prediction (multi-day horizon).

* 🔌 **Energy** – Electricity demand forecasting for smart grids.

* 🛒 **Retail** – Product demand prediction for inventory optimization.

* 🏥 **Healthcare** – Patient admission forecasting with seasonal trends.

---

**🛠️ Requirements**

* Python 3.8+

* Streamlit

* PyTorch

* Pandas, scikit-learn, statsmodels, Plotly

Install dependencies:

```bash
pip install -r requirements.txt
```

---

**🚀 Getting Started**

```bash
git clone https://github.com/Nyx1311/TimeLSTM.git
cd TimeLSTM

pip install -r requirements.txt

streamlit run app.py
```

---

**🤝 Contributing**

Pull requests are welcome!
For major changes, open an issue first to discuss improvements.

---

**📜 License**

This project is licensed under the **MIT License**.

---
Test app at https://timelstm.streamlit.app/
