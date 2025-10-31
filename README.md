# 🧠 AI-Powered Network Intrusion Detection System (NIDS)

An AI-based Network Intrusion Detection System with a Streamlit-powered GUI that intelligently detects malicious network traffic using machine learning models. It analyzes network flow data and helps identify anomalies or attacks in real-time. Designed for cybersecurity learners, analysts, and professionals aiming to automate network monitoring.

---

## 🚀 Features

- **AI-driven Detection:** Uses supervised and unsupervised learning to classify and detect network anomalies.  
- **Smart Mode Selection:** Automatically adapts based on dataset structure (with or without label columns).  
- **Interactive GUI:** Built using Streamlit for real-time data visualization and analysis.  
- **Flexible Input Support:** Upload network flow files in CSV or Excel formats.  
- **Auto Data Preprocessing:** Automatically cleans, encodes, and standardizes features.  
- **Explainable AI:** SHAP visualization for understanding model predictions.  
- **Result Export:** Download analyzed results in CSV format.  

---

## 🧰 Tools & Technologies Used

- **Python 3.8+**
- **Streamlit** — for building interactive dashboards  
- **Pandas** — for data manipulation  
- **NumPy** — for numerical computation  
- **Scikit-learn** — for training and evaluating ML models  
- **Joblib** — for saving/loading trained models  
- **SHAP** — for explainable AI visualization  
- **Matplotlib & Seaborn** — for data visualization  

---

## ⚙️ How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-nids-streamlit.git
cd ai-nids-streamlit
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)
If you want to train a new model using your dataset:
```bash
python train_model.py
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

### 5. Use the Application
- Upload your **network dataset** (CSV or Excel).  
- Select **Supervised** (requires label column) or **Unsupervised** mode.  
- View predictions, visualize anomalies, and download the processed results.  

---

## 🧑‍💻 Author

**Nithin1734**  
Cybersecurity Enthusiast 

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this project with proper credit.

---

## 🤝 Contributing

Contributions are highly appreciated!  
If you’d like to enhance model accuracy, improve preprocessing, or upgrade the UI, fork this repository and submit a pull request to collaborate on making this project better.
