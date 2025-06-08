# LSTM for Backorder Prediction - Supply Chain Forecasting

A sophisticated LSTM model for predicting backorders in supply chain management, featuring synthetic time-series transformation of tabular data.

## Key Features

- **Synthetic Time-Series Conversion**: Transforms tabular data into sequential format
- **Advanced LSTM Architecture**: 2-layer LSTM with BatchNorm and Dropout
- **Class Imbalance Handling**: Custom class weighting and evaluation metrics
- **Full Training Pipeline**: From data loading to model evaluation
- **Production-Ready**: Model checkpointing and early stopping

## Dataset Requirements

Preprocessed CSV files from the data cleaning pipeline:
- `Train_Preprocess.csv`
- `Test_Preprocess.csv`

## Installation

```bash
git clone https://github.com/yourusername/backorder-lstm.git
cd backorder-lstm
pip install -r requirements.txt
