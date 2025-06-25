# Advanced Stock Market Time Series Forecasting & Dashboard Suite

## 🚀 Project Overview
This repository delivers a professional, end-to-end solution for multi-stock time series forecasting, combining advanced statistical and deep learning models with interactive dashboards. Designed for group projects, business users, and data scientists, it enables robust forecasting, explainability, and seamless integration with Power BI and Streamlit.

---

## ✨ Features
- **Multi-stock, multi-year data** from Yahoo Finance (10+ years, US & global indices)
- **Automated data download, cleaning, and feature engineering** (SMA, EMA, RSI, MACD, Bollinger Bands, Volatility)
- **Forecasting models:**
  - ARIMA (statistical)
  - SARIMA (seasonal statistical)
  - LSTM (deep learning, Keras/TensorFlow)
- **Multi-horizon forecasts:** 7, 30, 90, 180 days
- **Batch processing for all stocks** (no manual steps required)
- **Professional, mobile-friendly Streamlit app**
  - Model selection, company/ticker mapping, technical indicator plots
  - Downloadable CSVs and reports
  - Light/dark mode, modern UI, HR-ready
- **Power BI dashboard-ready outputs** (CSV for all models/horizons)
- **Easy extensibility:** add new stocks, models, or features

---

## 🗂️ Project Structure
```
Time-Series-Analysis/
│
├── data/                # Raw & processed data, model outputs, plots
│   ├── model_outputs/   # Forecast CSVs & plots for each stock/model/horizon
│   └── eda_plots/       # Technical indicator plots
├── src/                 # Python scripts for data, features, modeling
│   ├── download_and_eda.py
│   ├── feature_engineering.py
│   ├── model_arima_prophet.py
│   ├── model_sarima.py
│   ├── model_lstm.py
│   └── merge_model_results.py
├── streamlit_app/       # Streamlit web app (app.py)
├── dashboard/           # (Optional) Power BI dashboard files
├── requirements.txt     # Python dependencies
├── README.md            # This documentation
└── .gitignore
```

---

## ⚡ Quickstart: Run the Full Pipeline
1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/Time-Series-Analysis.git
   cd Time-Series-Analysis
   ```
2. **Install Python 3.11+** (recommended for TensorFlow compatibility)
3. **Create and activate a virtual environment** (optional but recommended)
   ```sh
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```
4. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
5. **Download and preprocess data**
   ```sh
   python src/download_and_eda.py
   python src/feature_engineering.py
   ```
6. **Run forecasting models**
   - ARIMA & Prophet: `python src/model_arima_prophet.py`
   - SARIMA: `python src/model_sarima.py`
   - LSTM: `python src/model_lstm.py`
7. **Merge model results for dashboards**
   ```sh
   python src/merge_model_results.py
   ```
8. **Launch the Streamlit app**
   ```sh
   streamlit run streamlit_app/app.py
   ```
9. **(Optional) Open Power BI dashboard**
   - Use the CSVs in `data/model_outputs/` for your Power BI reports.

---

## 🖥️ Usage Details
- **Streamlit App:**
  - Select stock, model, and forecast horizon
  - Visualize technical indicators and forecasts
  - Download results as CSV
  - Mobile and desktop friendly
- **Power BI:**
  - Import any `*_all_models_results_*.csv` for multi-model, multi-horizon analysis
- **Customizing Stocks/Models:**
  - Edit the `tickers` list in `src/download_and_eda.py` and rerun the pipeline
  - Add new models by creating scripts in `src/` and updating the merge script

---

## 🛠️ Troubleshooting & Tips
- **TensorFlow not found?**
  - Ensure you're using the same Python environment for both `pip install` and running scripts
  - On Windows, use the full path to Python if needed:  
    `& "C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe" src/model_lstm.py`
- **Power BI CSV import:**
  - Use the merged CSVs for best results
- **Streamlit port in use?**
  - Run with `streamlit run streamlit_app/app.py --server.port 8502`
- **Add/Remove stocks:**
  - Edit the `tickers` list in `src/download_and_eda.py`
- **Model errors?**
  - Check data completeness and feature engineering outputs

---

## 🤝 Contribution Guide
- Fork the repo and create a feature branch
- Follow PEP8 and add docstrings/comments
- Test your scripts on a fresh environment
- Submit a pull request with a clear description

---

## 👥 Team & Credits
- Project Lead: Vishal Deep
- Contributors: Vishal Deep, Vaibhav Sharma, Rishi, Keval Rathod, Shaion Sanyai
- Special thanks: Open source libraries (Yahoo Finance, pandas, scikit-learn, TensorFlow, Streamlit, etc.)

---

## 📄 License
MIT

---

## 📬 Contact
For questions, issues, or collaboration, open an issue or contact [vishalyep1022@gmail.com]. 
