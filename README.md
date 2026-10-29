# Customer Churn Prediction (Subscription Streaming Service)

A Streamlit app that predicts whether a customer is likely to churn, using a trained scikit-learn classification pipeline.

> **Use case:** a Netflix-inspired subscription streaming scenario. This project is **not** affiliated with Netflix, and Netflix does not use this model.

> **Data note:** no real dataset was supplied, so the project uses a **synthetic dataset** (4,000 customers) created by `generate_data.py`. Churn depends on behaviour (payment failures, complaints, watch hours, logins, tenure, auto-renewal) plus random noise, so results are realistic but not perfect. To use real data, place a CSV with the same columns at `data/customer_churn.csv` and re-run training.

## How it works
```
Dataset -> cleaning -> ColumnTransformer (impute + scale + one-hot) -> stratified split
-> train Logistic Regression / Decision Tree / Random Forest -> evaluate
-> save best full pipeline (model/churn_model.pkl) + model/metrics.json
-> Streamlit app loads pipeline -> user input DataFrame -> predict() + predict_proba()
-> YES/NO + churn probability
```
The saved pipeline contains the preprocessing and the model together, so Streamlit inputs get exactly the same preprocessing as training. The app never retrains.

## Run
```bash
pip install -r requirements.txt
python generate_data.py     # optional; train_model.py creates data if missing
python train_model.py       # trains, compares, saves model/
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Builds the synthetic dataset |
| `train_model.py` | Validates, cleans, trains 3 models, evaluates, saves best |
| `app.py` | Streamlit UI |
| `model/churn_model.pkl` | Saved preprocessing + model pipeline |
| `model/metrics.json` | Metrics for the selected and all models |

## Model selection
All three models are scored on the same held-out 20% test set (Accuracy, Precision, Recall, F1, ROC-AUC). The final model is the one with the highest ROC-AUC (F1 breaks ties) - chosen from real results, not fixed in advance.

## Notes
- Risk level (LOW < 30%, MEDIUM 30-60%, HIGH > 60%) is only a display layer over the model's probability.
- The "Potential contributing factors" section is simple rule-based commentary on the entered values. It does not decide or explain the model's prediction.
