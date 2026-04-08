{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ea4fe7bc-7183-4774-85c3-6bcbca7f1e34",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.55\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.00      0.00      0.00         1\n",
      "           1       0.42      0.83      0.56         6\n",
      "           3       0.75      0.46      0.57        13\n",
      "\n",
      "    accuracy                           0.55        20\n",
      "   macro avg       0.39      0.43      0.38        20\n",
      "weighted avg       0.61      0.55      0.54        20\n",
      "\n",
      "Model and scaler saved successfully!\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "E:\\Users\\anaconda\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", result.shape[0])\n",
      "E:\\Users\\anaconda\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", result.shape[0])\n",
      "E:\\Users\\anaconda\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, f\"{metric.capitalize()} is\", result.shape[0])\n"
     ]
    }
   ],
   "source": [
    "# train_model.py\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "import joblib\n",
    "\n",
    "# =========================\n",
    "# LOAD DATA\n",
    "# =========================\n",
    "df = pd.read_csv(\"clean_marriage_data.csv\")\n",
    "\n",
    "# =========================\n",
    "# DATA CLEANING\n",
    "# =========================\n",
    "# drop missing values\n",
    "df = df.dropna()\n",
    "\n",
    "# remove duplicates\n",
    "df = df.drop_duplicates()\n",
    "\n",
    "# =========================\n",
    "# ENCODING CATEGORICAL DATA\n",
    "# =========================\n",
    "le = LabelEncoder()\n",
    "\n",
    "categorical_cols = [\"helping\", \"fam_id\", \"parent_supp\", \"boy_finan\", \"HARUSI\"]\n",
    "\n",
    "for col in categorical_cols:\n",
    "    df[col] = le.fit_transform(df[col])\n",
    "\n",
    "# =========================\n",
    "# FEATURE & TARGET\n",
    "# =========================\n",
    "X = df.drop(\"HARUSI\", axis=1)\n",
    "y = df[\"HARUSI\"]\n",
    "\n",
    "# =========================\n",
    "# FEATURE SCALING\n",
    "# =========================\n",
    "scaler = StandardScaler()\n",
    "X = scaler.fit_transform(X)\n",
    "\n",
    "# =========================\n",
    "# TRAIN / TEST SPLIT\n",
    "# =========================\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "# =========================\n",
    "# MODEL TRAINING\n",
    "# =========================\n",
    "model = LogisticRegression()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# =========================\n",
    "# EVALUATION\n",
    "# =========================\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))\n",
    "\n",
    "# =========================\n",
    "# SAVE MODEL\n",
    "# =========================\n",
    "joblib.dump(model, \"marriage_model.pkl\")\n",
    "joblib.dump(scaler, \"scaler.pkl\")\n",
    "\n",
    "print(\"Model and scaler saved successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7692816e-b9dc-4386-93ef-261f3d4caa9f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb93da7e-53d2-476e-97da-1288b97bbcef",
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
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
