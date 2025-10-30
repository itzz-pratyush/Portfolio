import os
import random
import math
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from math import pi

warnings.filterwarnings("ignore")
tqdm.pandas()
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Streamlit page config
st.set_page_config(page_title="NIFTY-500 AI Portfolio Builder", layout="wide")
st.title("📈 NIFTY-500 AI Portfolio Builder")
st.write("Optimize your investment portfolio using AI-driven predictions on NIFTY-500 stocks.")

# ================================================================
# Helper Functions (from your original script, slightly modified)
# ================================================================

@st.cache_data  # Cache to avoid re-fetching on every run
def get_nifty500_tickers():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        df = pd.read_csv(url)
        tickers = [t.strip() + ".NS" for t in df["Symbol"].dropna().unique().tolist()]
        return tickers
    except Exception as e:
        st.warning(f"Failed to fetch official list: {e}")
        return []

@st.cache_data
def validate_tickers_yahoo(tickers):
    valid = []
    for t in tqdm(tickers, desc="Validating tickers"):
        try:
            df = yf.download(t, period="5d", interval="1d", progress=False)
            if not df.empty:
                valid.append(t)
        except Exception:
            continue
    return valid

@st.cache_data
def fetch_price_data(tickers, period="1y", interval="1d", retries=3):
    all_data, failed = {}, []
    batch_size = 50
    for i in tqdm(range(0, len(tickers), batch_size), desc="Fetching stock data"):
        batch = tickers[i:i+batch_size]
        success = False
        for _ in range(retries):
            try:
                data = yf.download(batch, period=period, interval=interval, group_by='ticker', progress=False)
                success = True
                break
            except Exception:
                continue
        if not success:
            failed.extend(batch)
            continue
        for t in batch:
            try:
                df = data[t] if t in data else None
                if df is not None and not df.empty:
                    df = df.copy().reset_index()
                    df.columns = [c.lower() for c in df.columns]
                    df["ticker"] = t
                    all_data[t] = df
                else:
                    failed.append(t)
            except Exception:
                failed.append(t)
    return all_data, failed

def build_features(df):
    df = df.copy()
    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["vol_5d"] = df["close"].rolling(5).std()
    df["rsi"] = 100 - (100 / (1 + (df["ret_1d"].rolling(14).apply(
        lambda x: (x[x>0].sum() / -x[x<0].sum()) if -x[x<0].sum()!=0 else 0))))
    df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)
    return df.dropna().reset_index(drop=True)

def build_feature_rows_from_price_data(price_data):
    rows = []
    for t, df in price_data.items():
        fdf = build_features(df)
        if not fdf.empty:
            rows.append(fdf)
    if not rows:
        return pd.DataFrame(), []
    return pd.concat(rows, ignore_index=True), []

def train_val_split(df, features):
    X = df[features].fillna(0).values
    y = df["target"].values
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

def train_anova_rf(df):
    features = ['ret_1d','ret_5d','ma_5','ma_20','vol_5d','rsi']
    X_train, X_val, y_train, y_val = train_val_split(df, features)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    selector = SelectKBest(f_classif, k=min(5, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X_train_sel, y_train)
    X_val_sel = selector.transform(X_val)
    preds = rf.predict(X_val_sel)
    prob = rf.predict_proba(X_val_sel)[:,1]
    metrics = {
        "accuracy": accuracy_score(y_val,preds),
        "precision": precision_score(y_val,preds),
        "recall": recall_score(y_val,preds),
        "f1": f1_score(y_val,preds),
        "roc_auc": roc_auc_score(y_val,prob)
    }
    return rf, features, selector, metrics

def make_lstm(input_shape):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def prepare_lstm_sequences(df, feats, lookback=5):
    seqX, seqY = [], []
    for _,grp in df.groupby('ticker'):
        g = grp.sort_values('date')
        X = g[feats].fillna(0).values
        y = g['target'].values
        for i in range(len(X)-lookback):
            seqX.append(X[i:i+lookback])
            seqY.append(y[i+lookback])
    return np.array(seqX), np.array(seqY)

def train_lstm(df):
    feats = ['ret_1d','ret_5d','ma_5','ma_20','vol_5d','rsi']
    X,y = prepare_lstm_sequences(df,feats)
    if len(X)<20:
        raise RuntimeError("Not enough samples for LSTM")
    Xtr,Xv,ytr,yv = train_test_split(X,y,test_size=0.2,random_state=SEED)
    m = make_lstm((X.shape[1],X.shape[2]))
    m.fit(Xtr,ytr,epochs=5,batch_size=32,verbose=0)
    pv = (m.predict(Xv,verbose=0).ravel()>0.5).astype(int)
    prob = m.predict(Xv,verbose=0).ravel()
    return m, feats, {
        "accuracy": accuracy_score(yv,pv),
        "precision": precision_score(yv,pv),
        "recall": recall_score(yv,pv),
        "f1": f1_score(yv,pv),
        "roc_auc": roc_auc_score(yv,prob)
    }

def allocate_portfolio(total, df):
    n=len(df)
    equal = total/n
    alloc=[]
    for _,r in df.iterrows():
        qty = math.floor(equal / r['last_close'])
        used = qty*r['last_close']
        alloc.append([r['ticker'], r['score'], r['last_close'], qty, used])
    port = pd.DataFrame(alloc, columns=['ticker','score','last_close','quantity','used_amount'])
    port['weight'] = port['used_amount']/port['used_amount'].sum()
    return port

# ================================================================
# Main App Logic
# ================================================================

# User Inputs
st.sidebar.header("Inputs")
investment_amount = st.sidebar.number_input("Total Investment Amount (e.g., 100000)", min_value=1000.0, value=100000.0, step=1000.0)
model_choice = st.sidebar.selectbox("Choose Model", ["ANOVA+RF", "LSTM"])
period = st.sidebar.selectbox("Data Period", ["6mo", "1y"], index=1)
run_button = st.sidebar.button("Run Portfolio Optimization")

if run_button:
    with st.spinner("Fetching and validating stocks..."):
        TICKERS_ALL = get_nifty500_tickers()
        VALID_TICKERS = validate_tickers_yahoo(TICKERS_ALL)
        if len(VALID_TICKERS) == 0:
            VALID_TICKERS = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","ITC.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","HINDUNILVR.NS","AXISBANK.NS","BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS","ULTRACEMCO.NS","WIPRO.NS","TITAN.NS","HCLTECH.NS"]
            st.warning("Using fallback tickers.")

    with st.spinner("Training models on fetched stocks..."):
        sample = VALID_TICKERS[:min(100, len(VALID_TICKERS))]
        sample_data, _ = fetch_price_data(sample, period=period)
        sample_df, _ = build_feature_rows_from_price_data(sample_data)
        rf_model, rf_feats, rf_selector, rf_metrics = train_anova_rf(sample_df)
        try:
            lstm_model, lstm_feats, lstm_metrics = train_lstm(sample_df)
        except:
            lstm_model, lstm_feats, lstm_metrics = None, [], {}

    # Model Comparison Visualizations
    st.header("Model Performance Comparison")
    metrics_df = pd.DataFrame([{**rf_metrics, 'model': 'ANOVA+RF'}, {**lstm_metrics, 'model': 'LSTM'}])
    fig, ax = plt.subplots(figsize=(10,6))
    metrics_melted = metrics_df.melt(id_vars='model', var_name='Metric', value_name='Score')
    sns.barplot(x='Metric', y='Score', hue='model', data=metrics_melted, palette="Set2", ax=ax)
    ax.set_ylim(0,1)
    st.pyplot(fig)

    # Choose Model
    chosen_type = 'anova_rf' if model_choice == "ANOVA+RF" else 'lstm'
    chosen_model = rf_model if chosen_type == 'anova_rf' else lstm_model
    chosen_feats = rf_feats if chosen_type == 'anova_rf' else lstm_feats
    chosen_selector = rf_selector if chosen_type == 'anova_rf' else None

    with st.spinner("Retraining on full data and scoring..."):
        full_data, _ = fetch_price_data(VALID_TICKERS, period=period)
        full_df, _ = build_feature_rows_from_price_data(full_data)
        if chosen_type == 'anova_rf':
            chosen_model, _, chosen_selector, _ = train_anova_rf(full_df)
        else:
            chosen_model, _, _ = train_lstm(full_df)

        scoring_df, _ = build_feature_rows_from_price_data(full_data)
        if chosen_type == 'anova_rf':
            X = scoring_df[chosen_feats].fillna(0).values
            X_sel = chosen_selector.transform(X)
            scoring_df['score'] = chosen_model.predict_proba(X_sel)[:,1]
        else:
            preds = []
            for t, g in tqdm(scoring_df.groupby('ticker'), desc="Scoring LSTM"):
                arr = g[chosen_feats].fillna(0).values
                if len(arr) < 5: continue
                seqs = np.array([arr[i:i+5] for i in range(len(arr)-5)])
                p = chosen_model.predict(seqs, verbose=0).ravel()
                preds.extend([(t, v) for v in p])
            pred_df = pd.DataFrame(preds, columns=['ticker','score'])
            scoring_df = scoring_df.merge(pred_df, on='ticker', how='left')

        latest = scoring_df.sort_values('date').groupby('ticker').tail(1)
        latest = latest[['ticker','score','close']].rename(columns={'close':'last_close'})
        latest = latest.sort_values('score', ascending=False).reset_index(drop=True)

    # Portfolio
    TOP_K = 15
    top = latest.head(TOP_K)
    port = allocate_portfolio(investment_amount, top)
    used = port['used_amount'].sum()
    leftover = investment_amount - used

    st.header("Portfolio Results")
    st.write(f"**Chosen Model:** {model_choice}")
    st.write(f"**Top {TOP_K} Stocks:**")
    st.dataframe(top)
    st.write(f"**Portfolio Allocation:** Total: {investment_amount:.2f}, Used: {used:.2f}, Leftover: {leftover:.2f}")
    st.dataframe(port)

    # Downloads
    st.download_button("Download Top 15 Allocation CSV", port.to_csv(index=False), "top15_allocation.csv", "text/csv")
    st.download_button("Download All Scored Tickers CSV", latest.to_csv(index=False), "scored_all_tickers.csv", "text/csv")

    # Visualizations
    st.header("Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.barplot(x='ticker', y='score', data=top, palette="viridis", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.pie(port['used_amount'], labels=port['ticker'], autopct='%1.1f%%', startangle=140)
        st.pyplot(fig)

    # Additional Charts
    fig, ax = plt.subplots(figsize=(12,6))
    sns.histplot(latest['score'], bins=30, kde=True, color='skyblue', ax=ax)
    st.pyplot(fig)

else:
    st.write("Enter inputs and click 'Run Portfolio Optimization' to start.")

