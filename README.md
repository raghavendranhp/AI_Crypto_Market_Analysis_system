# AI Crypto Market Signal & Risk Analyzer

An end-to-end Machine Learning and LLM-powered project to analyze cryptocurrency market trends, detect anomalies, and generate actionable trading signals using Groq's Llama 3.1 model.

## Folder Structure
```text
/
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── app/
│   ├── app.py
│   ├── llm_insights.py
│   └── models/
│       ├── random_forest_trend.joblib
│       └── isolation_forest_anomaly.joblib
├── prompts/
│   └── system_prompt.txt
├── data/
│   └── crypto_trading_dataset.csv
├── .env
├── .instructions
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Configure the `.env` File
Create a `.env` file in the root directory (if not already created) and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Install Dependencies
Ensure you have Python installed. Create a virtual environment and install the required packages:
```bash
python -m venv venv
# On Windows: venv\\Scripts\\activate
# On Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate Models
If you need to regenerate the models from the mock dataset, run:
```bash
python train_and_generate.py
```
Or execute the Jupyter Notebook located at `notebooks/01_eda_and_modeling.ipynb`.

### 4. Run the Streamlit Dashboard
To launch the interactive dashboard, run the following command from the root directory:
```bash
streamlit run app/app.py
```

## Architecture
- **Phase 1**: Data loading, feature engineering (MA7, MA30, Volatility Index, Volume Spike Ratio), and model training using `RandomForestClassifier` (Trend Classification) and `IsolationForest` (Anomaly Detection).
- **Phase 2 & 3**: Integration with the `Groq` API to leverage state-of-the-art Llama 3.1 LLM for concise, expert quantitative trading insights. The LLM adheres strictly to a 500-character limit and avoids emojis.
- **Phase 4**: An interactive `Streamlit` frontend to perform What-If analysis by adjusting market parameters dynamically. The final output is rendered strictly as robust JSON.
