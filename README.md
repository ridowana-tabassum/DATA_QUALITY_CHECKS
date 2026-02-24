🔗 **Live App:** https://dataqualitychecks-ybliu97awfwtbfzswamacj.streamlit.app/

---

# 📊 Financial Data Quality Framework

An automated data quality pipeline for validating **company financial revenue data** extracted from multiple sources.
This project combines **rule-based validation and LLM-powered anomaly detection** to identify errors, inconsistencies, and plausibility risks before data ingestion.

---

## 🚀 Overview

Financial datasets from APIs, scrapers, or multiple providers often contain:

* Missing values
* Currency mismatches
* Outliers or unit errors
* Duplicate or inconsistent records
* Contextual anomalies that rules alone cannot detect

This framework provides a **scalable, automated solution** to detect such issues and support reliable downstream analytics.

The system runs **5 dimensions of quality checks**, flags problematic records, and generates structured outputs for decision-making.

---

## ⚙️ Features

### ✅ Automated Validation

* Completeness
* Validity
* Consistency
* Uniqueness
* LLM-based Plausibility

### 🧠 Hybrid Approach

* Rule-based checks for structured validation
* Optional LLM reasoning for contextual anomaly detection

### 📈 Interactive UI

* Built with Streamlit
* Configurable thresholds
* Upload and validate datasets
* Visual inspection of flagged records

### 📂 Multiple Output Formats

* Flagged dataset
* JSON quality report
* Filtered issue export

---

## 🏗️ Architecture

The pipeline follows a modular structure:

```
Input Dataset  
   ↓  
Load & Parse  
   ↓  
Quality Checks  
   ↓  
Row-Level Flags  
   ↓  
Export Results  
```

Each dimension produces structured issues with:

* Severity
* Description
* Affected rows
* Recommended actions

---

## 🔍 Quality Checks

### 1. Completeness

Detects:

* Missing company names
* Missing revenue for active public companies
* Dataset-level null thresholds

### 2. Validity

Validates:

* Schema and required columns
* Revenue sanity limits
* Fiscal period formatting

### 3. Consistency

Identifies:

* Extreme year-over-year changes
* Zero revenue in active companies
* Time-series gaps
* Repeated values

### 4. Uniqueness

Checks:

* Duplicate primary keys
* Exact duplicate rows
* Multiple IDs mapped to the same company

### 5. Plausibility (LLM-based)

Uses a language model to:

* Evaluate suspicious patterns
* Distinguish real business events from data errors
* Provide structured verdicts

---

## 🧾 Severity Levels

| Level       | Description                          |
| ----------- | ------------------------------------ |
| HARD_REJECT | Critical errors that block ingestion |
| SOFT_FLAG   | Suspicious data requiring review     |

---

## 📊 Example Use Cases

* Financial data platforms
* Market intelligence
* Data ingestion pipelines
* Due diligence workflows
* Analytics and reporting
* Benchmarking and research

---

## 🖥️ Streamlit App

The project includes an interactive UI where users can:

* Upload Excel or CSV datasets
* Configure validation thresholds
* Enable or disable checks
* Run quality analysis
* Explore flagged records
* Download structured outputs

---

## 📥 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/data-quality-framework.git
cd data-quality-framework
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🔑 Optional LLM Setup

To enable the plausibility check:

1. Generate an API key
2. Enter it in the Streamlit UI
3. Configure the number of candidate records

If no key is provided, the system will skip this step.

---

## 📂 Input Requirements

Supported formats:

* `.xlsx`
* `.xls`
* `.csv`

Required columns:

```
timevalue
providerkey
companynameofficial
fiscalperiodend
operationstatustype
ipostatustype
geonameen
industrycode
REVENUE
unit_REVENUE
```

---

## 📤 Outputs

The framework generates:

1. Flagged dataset
2. JSON report
3. Filtered issue export

These outputs support both automated pipelines and manual review.

---

## 📌 Key Insights

* Multi-source financial data introduces significant quality risk
* Completeness and consistency dominate structural issues
* LLM reasoning adds a complementary validation layer
* Automated validation is essential for scalable data ingestion
