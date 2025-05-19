{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9f0979ac-0545-437f-a816-7bf9cedf345d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import streamlit as st\n",
    "import shap\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "489a2023-597c-4932-891c-30be15da2d13",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from xgboost import XGBClassifier\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "1a95764d-39cb-438f-a5c4-f221d5630427",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Dataset\n",
    "df = pd.read_csv(\"diabetes_binary_health_indicators_BRFSS2021.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8aef819c-e7df-43d9-bb17-ff25b2bd157c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature Engineering\n",
    "df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=['<30', '30-45', '45-60', '60+'])\n",
    "df['BMIGroup'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])\n",
    "df['BP_Chol_Risk'] = df['HighBP'] + df['HighChol']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1d6a0780-0e12-4b8b-9092-322d5f60d2d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop rows with missing values in newly created columns\n",
    "df.dropna(subset=['AgeGroup', 'BMIGroup'], inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d044767c-491d-42b4-b0af-8677f90c0f9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Normalize continuous variables\n",
    "scaler = StandardScaler()\n",
    "df[['BMI', 'PhysHlth', 'MentHlth']] = scaler.fit_transform(df[['BMI', 'PhysHlth', 'MentHlth']])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "6921810b-b9e5-4fe6-a5f2-d26bdf195dc4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare data for modeling\n",
    "X = df.drop(columns=['Diabetes_binary', 'AgeGroup', 'BMIGroup'])\n",
    "y = df['Diabetes_binary']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "a5f4010f-9f78-42c1-af48-aaeda1aeedc7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split the dataset\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "e4cd36bf-f2cb-4a92-9b0f-83e0b8dce47d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize Models\n",
    "models = {\n",
    "    'Logistic Regression': LogisticRegression(),\n",
    "    'Random Forest': RandomForestClassifier(),\n",
    "    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "04d6f453-72d4-4850-8d7c-c7227d9d5062",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Logistic Regression Results:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "         0.0       0.88      0.98      0.92     40562\n",
      "         1.0       0.56      0.15      0.24      6714\n",
      "\n",
      "    accuracy                           0.86     47276\n",
      "   macro avg       0.72      0.57      0.58     47276\n",
      "weighted avg       0.83      0.86      0.83     47276\n",
      "\n",
      "ROC-AUC: 0.8170\n",
      "\n",
      "Random Forest Results:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "         0.0       0.88      0.97      0.92     40562\n",
      "         1.0       0.49      0.17      0.25      6714\n",
      "\n",
      "    accuracy                           0.86     47276\n",
      "   macro avg       0.68      0.57      0.58     47276\n",
      "weighted avg       0.82      0.86      0.83     47276\n",
      "\n",
      "ROC-AUC: 0.7921\n",
      "\n",
      "XGBoost Results:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "         0.0       0.87      0.98      0.92     40562\n",
      "         1.0       0.54      0.15      0.24      6714\n",
      "\n",
      "    accuracy                           0.86     47276\n",
      "   macro avg       0.71      0.57      0.58     47276\n",
      "weighted avg       0.83      0.86      0.83     47276\n",
      "\n",
      "ROC-AUC: 0.8207\n"
     ]
    }
   ],
   "source": [
    "# Evaluate models\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    y_pred = model.predict(X_test)\n",
    "    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])\n",
    "    print(f\"\\n{name} Results:\")\n",
    "    print(classification_report(y_test, y_pred))\n",
    "    print(f\"ROC-AUC: {roc_auc:.4f}\")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "536b4407-6330-4d6d-b080-31b8ffac203d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SHAP Explainability for XGBoost\n",
    "explainer = shap.Explainer(models['XGBoost'])\n",
    "shap_values = explainer(X_test)\n",
    "shap.summary_plot(shap_values, X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "c782423b-b954-42a2-ac0f-ee97d23b5288",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-09 16:21:20.129 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-09 16:21:20.131 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Streamlit UI\n",
    "st.title(\" Diabetes Risk Prediction\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "b3219dc4-199f-4902-ba3d-80e21cfe3457",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-09 16:20:22.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-09 16:20:22.785 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-09 16:20:22.786 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-09 16:20:22.787 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "st.write(\"Fill in the details below to check the diabetes risk\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f53c8be6-5c74-4185-84d9-950d505feca0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
