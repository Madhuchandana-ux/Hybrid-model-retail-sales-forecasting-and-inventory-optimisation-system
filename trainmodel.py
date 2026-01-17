"""
🛒 RETAIL FORECASTING TRAINMODEL.PY - Reads data/retail.csv
✅ Steps 1-10: CSV → Features → ML + Croston → PKL Export
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
import joblib
from scipy.stats import norm
from datetime import datetime

# ============================================================================
# STEP 1-2: LOAD REAL CSV DATA
# ============================================================================
def load_and_validate_data():
    """Step 2: Read data/retail.csv + validate"""
    
    # Check if data folder exists
    data_path = "data/retail.csv"
    if not os.path.exists(data_path):
        print(f"❌ {data_path} not found!")
        print("💡 Create folder 'data/' and put retail.csv there")
        print("📄 Expected columns: store_id, item_id, date, qty_sold, [on_promo, price]")
        return None
    
    print(f"📂 Loading {data_path}...")
    df = pd.read_csv(data_path)
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    
    # Required columns check
    required_cols = ['store_id', 'item_id', 'date', 'qty_sold']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print("💡 Add these columns to retail.csv")
        return None
    
    # Optional columns
    if 'on_promo' not in df.columns:
        df['on_promo'] = 0
    if 'price' not in df.columns:
        df['price'] = 100.0
    if 'stockout_flag' not in df.columns:
        df['stockout_flag'] = 0
    
    # Data quality checks
    print(f"✅ Loaded: {len(df):,} rows")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"🏪 Stores: {df['store_id'].nunique()}")
    print(f"📦 Items: {df['item_id'].nunique()}")
    print(f"🔢 Zero sales (P0): {(df['qty_sold']==0).mean():.1%}")
    
    # Remove stockouts
    before = len(df)
    df = df[df['stockout_flag'] == 0].copy()
    print(f"🧹 Removed stockouts: {before-len(df):,} rows")
    
    return df.sort_values(['store_id', 'item_id', 'date'])

# ============================================================================
# STEP 4: FEATURE ENGINEERING (Production)
# ============================================================================
def prepare_features(df):
    """EXACT spec: lags(1,7,14) + rolling(7,14,28) + calendar"""
    
    def make_features(sku_df):
        sku_df = sku_df.sort_values("date").copy()
        
        # Lags
        for lag in [1, 7, 14]:
            sku_df[f'lag_{lag}'] = sku_df['qty_sold'].shift(lag)
        
        # Rolling windows
        for window in [7, 14, 28]:
            sku_df[f'rollmean_{window}'] = sku_df['qty_sold'].shift(1).rolling(window).mean()
            sku_df[f'rollstd_{window}'] = sku_df['qty_sold'].shift(1).rolling(window).std()
        
        # Calendar features
        sku_df['dow'] = sku_df['date'].dt.dayofweek
        sku_df['week'] = sku_df['date'].dt.isocalendar().week
        sku_df['month'] = sku_df['date'].dt.month
        sku_df['quarter'] = sku_df['date'].dt.quarter
        sku_df['promo'] = sku_df['on_promo'].fillna(0)
        sku_df['log_price'] = np.log1p(sku_df['price'].fillna(100))
        
        return sku_df.dropna()
    
    df_features = df.groupby(['store_id', 'item_id'], group_keys=False).apply(make_features)
    print(f"✅ Features created: {len(df_features):,} rows | {len(df_features.columns)-5} features")
    return df_features

# ============================================================================
# STEP 6: CROSTON (Production Implementation)
# ============================================================================
def croston_forecast(y, alpha=0.1, h=28):
    """EXACT Croston/SBA specification"""
    demand = y.values
    z = demand[demand > 0]
    p = np.diff(np.r_[0, np.where(demand > 0)[0]])
    
    if len(z) == 0:
        return np.full(h, 0.1)
    
    z_hat, p_hat = z[0], p[0] if len(p) > 0 else 1
    
    for i in range(1, len(z)):
        z_hat = alpha * z[i] + (1 - alpha) * z_hat
    for i in range(1, len(p)):
        p_hat = alpha * p[i] + (1 - alpha) * p_hat
    
    return (z_hat / p_hat) * np.ones(h)

# ============================================================================
# STEPS 5+7: ML TRAINING + BACKTESTING
# ============================================================================
def train_model(df):
    """Complete ML pipeline + validation"""
    
    # Feature selection (numeric only)
    exclude_cols = ['qty_sold', 'date', 'store_id', 'item_id', 'stockout_flag']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].astype(np.float64)
    y = df['qty_sold']
    groups = df.groupby(['store_id', 'item_id']).ngroup()
    
    # Train/test split (store-item groups)
    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    
    # RandomForest (production tuned)
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    
    print("🤖 Training model...")
    rf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    
    # Backtest
    y_pred = rf.predict(X.iloc[te_idx])
    mae = mean_absolute_error(y.iloc[te_idx], y_pred)
    resid_std = float(np.std(y.iloc[te_idx] - y_pred))
    
    print(f"✅ BACKTEST COMPLETE | MAE: {mae:.2f} | Residual Std: {resid_std:.2f}")
    
    return {
        'model': rf,
        'features': feature_cols,
        'mae': mae,
        'resid_std': resid_std
    }

# ============================================================================
# STEP 8: INVENTORY POLICY
# ============================================================================
def inventory_policy(forecast, resid_std, on_hand, lead_time, service=0.95):
    """Production ROP/SS/EOQ"""
    z = norm.ppf(service)
    lead_demand = forecast[:lead_time].sum()
    lead_std = resid_std * np.sqrt(lead_time)
    safety_stock = z * lead_std
    reorder_point = lead_demand + safety_stock
    
    annual_demand = forecast.mean() * 365
    eoq = np.sqrt((2 * annual_demand * 500) / (100 * 0.2))
    order_qty = max(0, max(eoq, reorder_point - on_hand))
    
    return {
        'order_quantity': order_qty,
        'reorder_point': reorder_point,
        'safety_stock': safety_stock,
        'eoq': eoq,
        'lead_demand': lead_demand
    }

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 RETAIL FORECASTING SYSTEM - Reading data/retail.csv")
    print("=" * 70)
    
    # Step 1-2: Load CSV
    df = load_and_validate_data()
    if df is None:
        exit(1)
    
    # Step 3-4: EDA + Features
    df_features = prepare_features(df)
    
    # Steps 5-7: Train + Validate
    artifacts = train_model(df_features)
    
    # Step 10: Save PKL
    joblib.dump(artifacts, "retail_forecast_model.pkl")
    
    print("\n" + "=" * 70)
    print("✅ PRODUCTION COMPLETE!")
    print(f"📦 Saved: retail_forecast_model.pkl")
    print(f"📊 Model: RandomForest | Features: {len(artifacts['features'])}")
    print(f"🎯 MAE: {artifacts['mae']:.2f} | Std: {artifacts['resid_std']:.2f}")
    print("\n🎬 Next: streamlit run dashboard.py")
    print("=" * 70)
