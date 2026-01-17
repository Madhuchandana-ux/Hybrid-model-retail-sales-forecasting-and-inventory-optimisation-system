# Hybrid-model-retail-sales-forecasting-and-inventory-optimisation-system
 Built an end-to-end system that forecasts item-level sales (daily/weekly) and translates forecasts into optimal replenishment decisions (safety stock, reorder points, EOQ), targeting stock-out reduction and working-capital efficiency.

**What it does**
Built a complete forecasting system that:

1.Predicts daily sales for store-item combinations

2.Handles intermittent demand (lots of zero sales days)

3.Generates purchase orders with reorder points and safety stock

4.Live dashboard for supply chain planners

**Tech Stack**
Data: pandas, numpy
ML: scikit-learn RandomForest
Intermittent: Croston/SBA method
Dashboard: Streamlit + Plotly

**Skills demonstrated**
1. Time series feature engineering (lags/rolling/calendar)
2. Hybrid ML + statistical modeling
3. Production dashboard (Streamlit)
4. Inventory optimization (ROP/SS/EOQ)
5. Data quality/validation

**Demo**
# 1. Clone repo
git clone Hybrid-model-retail-sales-forecasting-and-inventory-optimisation-system-repo

# 2. Run notebook (cells 1-9)
jupyter notebook notebook.ipynb

# 3. Launch dashboard
streamlit run dashboard.py
