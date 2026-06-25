# Shopper Spectrum: Customer Segmentation using RFM Analysis

## Overview

Shopper Spectrum is a customer segmentation project that analyzes e-commerce customer behavior using **RFM (Recency, Frequency, Monetary) Analysis** and **K-Means Clustering**.

The goal is to identify distinct customer groups and help businesses create targeted marketing strategies, improve customer retention, and maximize revenue.

---

## Features

* Customer segmentation using RFM metrics
* Data preprocessing and feature engineering
* K-Means clustering for customer grouping
* Interactive Streamlit dashboard
* Cluster visualization and insights
* Business recommendations for each customer segment

---

## Dataset

This project uses the **Olist E-commerce Dataset**, containing information about:

* Customers
* Orders
* Order Items
* Payments
* Products

The dataset is not included in this repository due to size limitations.

---

## Methodology

### 1. Data Cleaning & Preparation

* Handled missing values
* Merged multiple datasets
* Converted date columns
* Removed invalid records

### 2. RFM Analysis

**Recency (R):**
Days since the customer's last purchase.

**Frequency (F):**
Total number of purchases made by the customer.

**Monetary (M):**
Total amount spent by the customer.

### 3. Feature Scaling

* Standardization of RFM features

### 4. Customer Segmentation

* Applied K-Means Clustering
* Determined optimal number of clusters
* Generated customer groups based on purchasing behavior

---

## Customer Segments

| Cluster   | Description                          |
| --------- | ------------------------------------ |
| Cluster 0 | High-value customers                 |
| Cluster 1 | Inactive/Low-spending customers      |
| Cluster 2 | Recent customers with lower spending |
| Cluster 3 | Loyal repeat customers               |

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## Project Structure

```text
Shopper-Spectrum/
│
├── app.py
├── notebook.ipynb
├── requirements.txt
├── models/
├── screenshots/
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/PragatiMaurya16/Shopper-Spectrum.git
cd Shopper-Spectrum
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Business Impact

The generated customer segments can be used for:

* Personalized marketing campaigns
* Customer retention strategies
* Loyalty programs
* Churn prediction initiatives
* Revenue optimization

---

## Author

**Pragati Maurya**

GitHub: https://github.com/PragatiMaurya16

---

## Support  

If you found this project useful, consider giving it a star!
It helps others discover the project and supports future development.
