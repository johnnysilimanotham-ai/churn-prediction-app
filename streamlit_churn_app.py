"""
Customer Churn Prediction App - Streamlit Cloud Compatible
Save this as: streamlit_churn_app.py
"""

import re
import io
import json
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Churn Prediction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'production_model' not in st.session_state:
    st.session_state.production_model = None
if 'feature_schema' not in st.session_state:
    st.session_state.feature_schema = None

# ==================== HELPER FUNCTIONS ====================

BINARY_FEATURE_NAMES_RE = re.compile(
    r'^(credit[_\s]?card|active[_\s]?member|has[_\s]?cr[_\s]?card)$',
    flags=re.IGNORECASE
)

def _looks_like_binary(s: pd.Series) -> bool:
    s = s.dropna()
    if s.empty:
        return False
    if pd.api.types.is_numeric_dtype(s):
        vals = set(pd.unique(pd.to_numeric(s, errors='coerce')))
        return vals.issubset({0, 1})
    vals = set(s.astype(str).str.strip().str.lower().unique())
    return vals.issubset({'0','1','yes','no','true','false','y','n'})

def coerce_binary_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Convert binary *feature* columns to categorical ('No','Yes') so your
    ColumnTransformer treats them as categorical. Never changes the target.
    """
    out = df.copy()
    for c in out.columns:
        if c == target_col:
            continue
        if BINARY_FEATURE_NAMES_RE.search(c) or _looks_like_binary(out[c]):
            s = out[c]
            if pd.api.types.is_numeric_dtype(s):
                s = pd.to_numeric(s, errors='coerce').map({0: 'No', 1: 'Yes'})
            else:
                s = s.astype(str).str.strip().str.lower().map({
                    '0':'No','no':'No','false':'No','n':'No',
                    '1':'Yes','yes':'Yes','true':'Yes','y':'Yes'
                })
            out[c] = pd.Categorical(s, categories=['No','Yes'])
    return out

def normalize_target(df, target_col):
    """Normalize target to 0/1 format"""
    if target_col not in df.columns:
        return df
    
    s = df[target_col].copy()
    
    # If it's already numeric, just ensure it's 0/1
    if pd.api.types.is_numeric_dtype(s):
        df[target_col] = pd.to_numeric(s, errors='coerce').fillna(0).clip(0, 1).astype(int)
    # If it's object/string type
    elif s.dtype == "O":
        def to_bin(x):
            if x is None or (isinstance(x, float) and pd.isna(x)): 
                return 0
            xs = str(x).strip().lower()
            if xs in {"yes", "true", "1", "1.0", "y", "t"}: 
                return 1
            if xs in {"no", "false", "0", "0.0", "n", "f"}: 
                return 0
            return 0
        df[target_col] = s.map(to_bin).astype(int)
    else:
        df[target_col] = s.astype(int)
    
    return df

def clean_data(df, target_col):
    """Clean and prepare data"""
    df = df.copy()
    
    # Drop ID columns first
    id_cols = [c for c in df.columns if 'id' in c.lower()]
    if id_cols:
        df = df.drop(columns=id_cols)
    
    # Normalize target column
    df = normalize_target(df, target_col)
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target_col])
    
    # Drop duplicates
    df = df.drop_duplicates()

    # **Coerce binary features to categorical so the encoder sees them**
    df = coerce_binary_features(df, target_col)
    
    # Verify we have both classes
    unique_classes = df[target_col].unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Target column must have at least 2 classes. Found only: {unique_classes}")
    
    return df

def build_preprocessor(df, target_col):
    """Build preprocessing pipeline"""
    df = coerce_binary_features(df, target_col)

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    
    schema = {
        "target": target_col,
        "numeric_features": num_cols,
        "categorical_features": cat_cols
    }
    
    return preprocessor, schema

def train_models(df, target_col, test_size=0.2):
    """Train all models and return results"""
    df = df.copy()
    
    # Ensure target is int type
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    
    # Check for class balance before proceeding
    class_counts = y.value_counts()
    st.info(f"📊 Class distribution: {class_counts.to_dict()}")
    
    if len(class_counts) < 2:
        st.error(f"❌ Only found class {y.unique()[0]} in the data. Cannot train models.")
        st.stop()
    
    preprocessor, schema = build_preprocessor(df, target_col)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    models_def = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42, max_iter=1000)
    }
    
    results = []
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models_def.items()):
        status_text.text(f"Training {name}...")
        
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
        results.append(metrics)
        
        trained_models[name] = {
            "pipeline": pipe,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
        
        progress_bar.progress((idx + 1) / len(models_def))
    
    status_text.empty()
    progress_bar.empty()
    
    results_df = pd.DataFrame(results).set_index("model").sort_values("f1", ascending=False)
    
    return results_df, trained_models, schema, X_test

# ==================== MAIN APP ====================

st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Platform</h1>', unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.selectbox(
    "📍 Navigation",
    ["📤 Upload Data", "🧹 Data Cleaning", "🤖 Train Models", "📊 Model Comparison", "🔮 Make Predictions"]
)

# ==================== PAGE 1: UPLOAD DATA ====================
if page == "📤 Upload Data":
    st.header("📤 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with customer data",
        type=['csv'],
        help="Your dataset should include customer features and a churn column"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_raw = df
            
            st.success(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", int(df.isnull().sum().sum()))
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("📊 Column Information")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Non-Null Count": df.count().values,
                "Null Count": df.isnull().sum().values
            })
            st.dataframe(info_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# ==================== PAGE 2: DATA CLEANING ====================

elif page == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning & Preparation")
    
    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        df = st.session_state.df_raw.copy()
        
        st.markdown("Configure data cleaning operations to prepare your dataset for analysis and modeling.")
        st.markdown("---")
        
        # Dataset Info Card
        st.subheader("📊 Current Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            missing_total = df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_total:,}")
        
        st.markdown("---")
        
        # Target Column Selection
        st.subheader("🎯 Target Column Selection")
        lower_cols = [c.lower() for c in df.columns]
        default_target_idx = lower_cols.index('churn') if 'churn' in lower_cols else 0
        target_col = st.selectbox(
            "Select Target Column (Churn/Outcome)",
            options=df.columns.tolist(),
            index=default_target_idx,
            help="Select the column you want to predict"
        )
        
        st.markdown("---")
        
        # Column Selection for Cleaning
        st.subheader("📋 Select Columns to Clean")
        
        # Initialize selected columns in session state
        if 'selected_columns' not in st.session_state:
            st.session_state.selected_columns = df.columns.tolist()
        
        ctop1, ctop2 = st.columns([3, 1])
        with ctop1:
            st.markdown("Choose which columns to include in cleaning operations")
        with ctop2:
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Select All", key="select_all", use_container_width=True):
                    st.session_state.selected_columns = df.columns.tolist()
                    # sync checkbox states
                    for c in df.columns:
                        st.session_state[f"col_check_{c}"] = True
                    st.rerun()
            with col_btn2:
                if st.button("Clear All", key="clear_all", use_container_width=True):
                    st.session_state.selected_columns = []
                    # sync checkbox states
                    for c in df.columns:
                        st.session_state[f"col_check_{c}"] = False
                    st.rerun()
        
        # Display columns with data type and missing info
        st.markdown("##### Available Columns")
        # Show binary-looking feature cols as categorical in the UI (target remains numeric)
        df_preview = coerce_binary_features(df, target_col=target_col)
        
        for col in df.columns:
            col_container = st.container()
            with col_container:
                col_checkbox, col_info = st.columns([3, 1])
                
                with col_checkbox:
                    default_checked = st.session_state.get(f"col_check_{col}", col in st.session_state.selected_columns)
                    is_selected = st.checkbox(
                        col,
                        value=default_checked,
                        key=f"col_check_{col}"
                    )
                    
                    if is_selected and col not in st.session_state.selected_columns:
                        st.session_state.selected_columns.append(col)
                    elif not is_selected and col in st.session_state.selected_columns:
                        st.session_state.selected_columns.remove(col)
                        
        # Display target as categorical in the UI if it's binary, but keep it numeric for modeling
                    if col == target_col and _looks_like_binary(df[col]):
                        dtype_label = "📝 Categorical"
                    else:
                        series_preview = df_preview[col]
                        if pd.api.types.is_categorical_dtype(series_preview) or not pd.api.types.is_numeric_dtype(series_preview):
                            dtype_label = "📝 Categorical"
                        else:
                            dtype_label = "🔢 Numeric"
                    
                    missing_count = df[col].isnull().sum()
                    missing_pct = (missing_count / len(df)) * 100 if len(df) else 0
                    info_text = f"{dtype_label}"
                    if missing_count > 0:
                        info_text += f" | ⚠️ {missing_count} missing ({missing_pct:.1f}%)"
                    
                    st.caption(info_text)
                
                with col_info:
                    # Show sample values
                    sample_values = df[col].dropna().unique()[:3]
                    if len(sample_values) > 0:
                        st.caption(f"Sample: {', '.join(map(str, sample_values))}")
        
        st.info(f"✅ {len(st.session_state.selected_columns)} of {len(df.columns)} columns selected")
        
        st.markdown("---")
        
        # Cleaning Options
        st.subheader("⚙️ Cleaning Operations")
        
        # Initialize cleaning options in session state
        if 'cleaning_options' not in st.session_state:
            st.session_state.cleaning_options = {
                'handle_missing': False,
                'missing_method': 'mean',
                'encode_categorical': False,
                'encoding_method': 'onehot',
                'remove_outliers': False,
                'normalize_data': False,
                'remove_duplicates': False
            }
        
        # Handle Missing Values
        with st.expander("🔧 Handle Missing Values", expanded=True):
            handle_missing = st.checkbox(
                "Enable missing value handling",
                value=st.session_state.cleaning_options['handle_missing'],
                help="Fill or remove missing data points"
            )
            st.session_state.cleaning_options['handle_missing'] = handle_missing
            
            if handle_missing:
                missing_method = st.selectbox(
                    "Method",
                    options=['mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'remove'],
                    index=['mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'remove'].index(
                        st.session_state.cleaning_options['missing_method']
                    ),
                    help="Choose how to handle missing values"
                )
                st.session_state.cleaning_options['missing_method'] = missing_method
                
                # Show explanation
                method_explanations = {
                    'mean': "Replace with column average (numeric only)",
                    'median': "Replace with middle value (numeric only)",
                    'mode': "Replace with most frequent value",
                    'forward_fill': "Use previous valid value",
                    'backward_fill': "Use next valid value",
                    'remove': "Delete rows with missing values"
                }
                st.caption(f"ℹ️ {method_explanations[missing_method]}")
        
        # Encode Categorical Variables
        with st.expander("🏷️ Encode Categorical Variables"):
            encode_categorical = st.checkbox(
                "Enable categorical encoding",
                value=st.session_state.cleaning_options['encode_categorical'],
                help="Convert text categories to numbers"
            )
            st.session_state.cleaning_options['encode_categorical'] = encode_categorical
            
            if encode_categorical:
                encoding_method = st.selectbox(
                    "Encoding Method",
                    options=['onehot', 'label'],  # removed 'ordinal' for cleanliness
                    index=['onehot', 'label'].index(
                        st.session_state.cleaning_options['encoding_method']
                    ) if st.session_state.cleaning_options['encoding_method'] in ['onehot','label'] else 0,
                    help="Choose encoding strategy"
                )
                st.session_state.cleaning_options['encoding_method'] = encoding_method
                
                # Show explanation
                encoding_explanations = {
                    'onehot': "Create binary columns for each category (recommended for ML)",
                    'label': "Assign integer to each category (0, 1, 2, ...)"
                }
                st.caption(f"ℹ️ {encoding_explanations[encoding_method]}")
        
        # Remove Outliers
        with st.expander("📊 Remove Outliers"):
            remove_outliers = st.checkbox(
                "Enable outlier removal",
                value=st.session_state.cleaning_options['remove_outliers'],
                help="Detect and remove statistical outliers using IQR method"
            )
            st.session_state.cleaning_options['remove_outliers'] = remove_outliers
            
            if remove_outliers:
                st.caption("ℹ️ Uses IQR (Interquartile Range) method: removes values beyond 1.5×IQR from Q1/Q3")
        
        # Normalize Data
        with st.expander("⚖️ Normalize Data"):
            normalize_data = st.checkbox(
                "Enable data normalization",
                value=st.session_state.cleaning_options['normalize_data'],
                help="Scale numeric features to 0-1 range"
            )
            st.session_state.cleaning_options['normalize_data'] = normalize_data
            
            if normalize_data:
                st.caption("ℹ️ Scales all numeric features to range [0, 1] using Min-Max scaling")
        
        # Remove Duplicates
        with st.expander("🔁 Remove Duplicates"):
            remove_duplicates = st.checkbox(
                "Enable duplicate removal",
                value=st.session_state.cleaning_options['remove_duplicates'],
                help="Delete duplicate rows from dataset"
            )
            st.session_state.cleaning_options['remove_duplicates'] = remove_duplicates
            
            if remove_duplicates:
                duplicate_count = df.duplicated().sum()
                st.caption(f"ℹ️ Found {duplicate_count} duplicate rows in current dataset")
        
        st.markdown("---")
        
        # Apply Cleaning Button
        if st.button("🚀 Apply Cleaning Operations", type="primary", use_container_width=True):
            try:
                with st.spinner("Cleaning data... Please wait."):
                    # Filter to selected columns + ensure target included
                    cols_to_process = [c for c in st.session_state.selected_columns if c != target_col]
                    if target_col not in st.session_state.selected_columns:
                        cols_to_process.append(target_col)
                    
                    df_clean = df[cols_to_process].copy()

                    # Make target numeric 0/1; coerce binary features to categorical ('No','Yes')
                    df_clean = normalize_target(df_clean, target_col)
                    df_clean = coerce_binary_features(df_clean, target_col)

                    # Show original stats
                    st.info(f"📊 Original data: {len(df_clean)} rows, {len(df_clean.columns)} columns")
                    if target_col in df_clean.columns:
                        st.info(f"📊 Original '{target_col}' distribution: {df_clean[target_col].value_counts(dropna=False).to_dict()}")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 1. Remove Duplicates
                    if st.session_state.cleaning_options['remove_duplicates']:
                        status_text.text("Removing duplicates...")
                        before_dup = len(df_clean)
                        df_clean = df_clean.drop_duplicates()
                        st.success(f"✅ Removed {before_dup - len(df_clean)} duplicate rows")
                    progress_bar.progress(20)
                    
                    # 2. Handle Missing Values
                    if st.session_state.cleaning_options['handle_missing']:
                        status_text.text("Handling missing values...")
                        method = st.session_state.cleaning_options['missing_method']
                        
                        if method == 'remove':
                            before_miss = len(df_clean)
                            df_clean = df_clean.dropna()
                            st.success(f"✅ Removed {before_miss - len(df_clean)} rows with missing values")
                        else:
                            for col in df_clean.columns:
                                if df_clean[col].isnull().any():
                                    if pd.api.types.is_numeric_dtype(df_clean[col]) and col != target_col:
                                        if method == 'mean':
                                            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                                        elif method == 'median':
                                            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                                        elif method == 'mode':
                                            df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
                                    else:
                                        if method == 'mode':
                                            df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
                                        elif method == 'forward_fill':
                                            df_clean[col] = df_clean[col].fillna(method='ffill')
                                        elif method == 'backward_fill':
                                            df_clean[col] = df_clean[col].fillna(method='bfill')
                            
                            st.success(f"✅ Filled missing values using {method} method")
                    progress_bar.progress(40)
                    
                    # 3. Remove Outliers
                    if st.session_state.cleaning_options['remove_outliers']:
                        status_text.text("Removing outliers...")
                        before_out = len(df_clean)
                        
                        numeric_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != target_col]
                        for col in numeric_cols:
                            Q1 = df_clean[col].quantile(0.25)
                            Q3 = df_clean[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            df_clean = df_clean[
                                (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                            ]
                        
                        st.success(f"✅ Removed {before_out - len(df_clean)} outlier rows")
                    progress_bar.progress(60)
                    
                    # 4. Normalize Target Column (no-op if already 0/1)
                    status_text.text("Normalizing target column...")
                    df_clean = normalize_target(df_clean, target_col)
                    progress_bar.progress(70)
                    
                    # 5. Encode Categorical Variables
                    if st.session_state.cleaning_options['encode_categorical']:
                        status_text.text("Encoding categorical variables...")
                        method = st.session_state.cleaning_options['encoding_method']
                        
                        # include 'category' so binary features get encoded
                        cat_cols = list(df_clean.select_dtypes(include=['object', 'category']).columns)
                        cat_cols = [c for c in cat_cols if c != target_col]
                        
                        if method == 'onehot':
                            df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
                            st.success(f"✅ One-hot encoded {len(cat_cols)} categorical columns")
                        elif method == 'label':
                            from sklearn.preprocessing import LabelEncoder
                            for col in cat_cols:
                                le = LabelEncoder()
                                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                            st.success(f"✅ Label encoded {len(cat_cols)} categorical columns")
                    progress_bar.progress(80)
                    
                    # 6. Normalize Numeric Data
                    if st.session_state.cleaning_options['normalize_data']:
                        status_text.text("Normalizing numeric features...")
                        from sklearn.preprocessing import MinMaxScaler
                        
                        numeric_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != target_col]
                        
                        if len(numeric_cols) > 0:
                            scaler = MinMaxScaler()
                            df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
                            st.success(f"✅ Normalized {len(numeric_cols)} numeric columns")
                    progress_bar.progress(100)
                    
                    # Final validation
                    status_text.text("Validating cleaned data...")
                    
                    # Verify we have both classes in target
                    unique_classes = pd.Series(df_clean[target_col].dropna().unique())
                    if unique_classes.nunique() < 2:
                        raise ValueError(
                            f"After cleaning, target column only has {unique_classes.nunique()} class(es): {unique_classes.tolist()}. "
                            f"Need at least 2 classes for prediction."
                        )
                    
                    # Save cleaned data
                    st.session_state.df_clean = df_clean
                    st.session_state.target_col = target_col
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.balloons()
                    st.success("✅ Data cleaned successfully!")
                    
                    # Show cleaned target distribution
                    st.info(f"📊 Cleaned '{target_col}' distribution: {df_clean[target_col].value_counts(dropna=False).to_dict()}")
                
            except ValueError as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure your target column has both churned (1) and non-churned (0) customers after cleaning.")
                
                # Debug info
                with st.expander("🔍 Debug Information"):
                    st.write(f"Target column: {target_col}")
                    if 'df_clean' in locals() and target_col in df_clean.columns:
                        st.write(f"Unique values in target: {pd.Series(df_clean[target_col].unique()).tolist()}")
                        st.write(f"Value counts: {df_clean[target_col].value_counts(dropna=False)}")
                        st.write(f"Data type: {df_clean[target_col].dtype}")
                st.stop()
            
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.stop()
            
            # Display Results
            st.markdown("---")
            st.subheader("📈 Cleaning Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows Before", len(df))
                st.metric("Rows After", len(st.session_state.df_clean))
                st.metric("Rows Removed", len(df) - len(st.session_state.df_clean))
            with col2:
                st.metric("Columns Before", len(df.columns))
                st.metric("Columns After", len(st.session_state.df_clean.columns))
                st.metric("Missing Values", int(st.session_state.df_clean.isnull().sum().sum()))
            
            st.subheader("📊 Cleaned Data Preview")
            st.dataframe(st.session_state.df_clean.head(20), use_container_width=True)
            
            # Target Distribution Visualization
            if target_col in st.session_state.df_clean.columns:
                st.subheader("🎯 Target Distribution")
                churn_counts = st.session_state.df_clean[target_col].dropna().value_counts().sort_index()
                
                if len(churn_counts) > 0:
                    vcol1, vcol2 = st.columns([2, 1])
                    
                    with vcol1:
                        labels = ['No Churn' if i == 0 else 'Churn' for i in churn_counts.index]
                        fig = px.pie(
                            values=churn_counts.values,
                            names=labels,
                            title="Churn Distribution",
                            color_discrete_sequence=['#667eea', '#764ba2']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with vcol2:
                        st.metric("No Churn", int(churn_counts.get(0, 0)))
                        st.metric("Churn", int(churn_counts.get(1, 0)))
                        
                        # Calculate churn rate
                        churn_rate = (churn_counts.get(1, 0) / churn_counts.sum()) * 100
                        st.metric("Churn Rate", f"{churn_rate:.1f}%")
                else:
                    st.warning("⚠️ No valid churn data found in target column")

# ==================== PAGE 3: TRAIN MODELS ====================
elif page == "🤖 Train Models":
    st.header("🤖 Train Machine Learning Models")

    if st.session_state.df_clean is None:
        st.warning("⚠️ Please clean your data first!")
    else:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression

        df_clean = st.session_state.df_clean.copy()
        target_col = st.session_state.target_col

        # --- Configuration UI ---
        st.subheader("⚙️ Training Configuration")

        # Model selection (no defaults; empty state)
        selected_models = st.multiselect(
            "Select Algorithms to Train",
            options=["random-forest", "svm", "logistic-regression"],
            default=[],
            placeholder="Choose options"
        )
        
        balance_classes = st.checkbox("Handle class imbalance (recommended)", value=True)

        # ================== DATASET SAMPLING ==================
        st.subheader("📉 Dataset Sampling")
        sample_option = st.radio(
            "How much data to use:",
            ["Use all data", "Use a percentage", "Use a fixed number of rows"],
            index=0
        )
        
        df_to_use = df_clean
        if sample_option == "Use a percentage":
            perc = st.slider("Percentage of dataset to use", 10, 100, 100, step=10)
            if perc < 100:
                df_to_use = df_clean.sample(frac=perc/100, random_state=42)
            st.caption(f"Using {perc}% → {len(df_to_use):,} rows out of {len(df_clean):,}")
        elif sample_option == "Use a fixed number of rows":
            max_rows = len(df_clean)
            n_rows = st.number_input(
                "Number of rows to use",
                min_value=1, max_value=max_rows, value=max_rows, step=max(1, max_rows // 20)
            )
            if n_rows < max_rows:
                df_to_use = df_clean.sample(n=int(n_rows), random_state=42)
            st.caption(f"Using {len(df_to_use):,} rows out of {len(df_clean):,}")
        
        total_rows = len(df_to_use)

# ================== TRAIN / TEST SPLIT ==================
        st.subheader("Train/Test Split")
        
        train_pct = st.slider(
            "\u00A0",  # blank label so ? tooltip shows
            min_value=10,
            max_value=90,
            step=5,
            value=80,
            label_visibility="visible",
            help="Pick the TRAIN percentage. Bounds 10–90 ensure each split gets at least 10%. "
        )
        test_pct = 100 - train_pct
        test_size = test_pct / 100.0
        
        train_rows = int(round(total_rows * (train_pct / 100.0)))
        test_rows = total_rows - train_rows
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f"**Training: {train_pct}% ({train_rows:,} rows)**")
        with c_right:
            st.markdown(
                f"<div style='text-align:right'><strong>Testing: {test_pct}% ({test_rows:,} rows)</strong></div>",
                unsafe_allow_html=True
            )
        
        st.caption(
            f"Final dataset: {total_rows:,} rows • {df_to_use.shape[1]} columns • "
            f"Split → Train: {train_pct}% ({train_rows:,}), Test: {test_pct}% ({test_rows:,})"
        )

        # Map selection to estimators
        def make_estimator(key: str):
            if key == "random-forest":
                return RandomForestClassifier(
                    n_estimators=300, max_depth=None, random_state=42, n_jobs=-1,
                    class_weight="balanced" if balance_classes else None
                )
            if key == "svm":
                return SVC(kernel="rbf", probability=True, random_state=42,
                           class_weight="balanced" if balance_classes else None)
            if key == "logistic-regression":
                return LogisticRegression(
                    max_iter=1000, solver="lbfgs",
                    class_weight="balanced" if balance_classes else None
                )
            raise ValueError(f"Unknown model '{key}'")

        # Start training
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if not selected_models:
                st.error("Please select at least one algorithm.")
            else:
                try:
                    with st.spinner("Training in progress…"):
                        prog = st.progress(0)
                        status = st.empty()

                        # 0) Prep target & features
                        status.text("Preparing data…")
                        df_to_use = normalize_target(df_to_use.copy(), target_col)
                        df_to_use = coerce_binary_features(df_to_use, target_col)

                        y = df_to_use[target_col].astype(int)
                        X = df_to_use.drop(columns=[target_col])

                        if y.nunique() < 2:
                            st.error("❌ Target column only has one class after sampling. Cannot train.")
                            st.stop()

                        prog.progress(10)

                        # 1) Build preprocessor once
                        status.text("Building preprocessing pipeline…")
                        preprocessor, schema = build_preprocessor(df_to_use, target_col)
                        prog.progress(25)

                        # 2) Split
                        status.text("Splitting train/test…")
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )
                        prog.progress(40)

                        # 3) Train each selected model
                        results = []
                        trained_models = {}
                        n = len(selected_models)
                        for i, key in enumerate(selected_models, start=1):
                            status.text(f"Training {key.replace('-', ' ').title()}…")
                            estimator = make_estimator(key)
                            pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
                            pipe.fit(X_train, y_train)

                            y_pred = pipe.predict(X_test)
                            y_proba = None
                            if hasattr(pipe.named_steps["model"], "predict_proba"):
                                try:
                                    y_proba = pipe.predict_proba(X_test)[:, 1]
                                except Exception:
                                    y_proba = None

                            m = {
                                "model": key,
                                "accuracy": accuracy_score(y_test, y_pred),
                                "precision": precision_score(y_test, y_pred, zero_division=0),
                                "recall": recall_score(y_test, y_pred, zero_division=0),
                                "f1": f1_score(y_test, y_pred, zero_division=0),
                                "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None
                            }
                            results.append(m)
                            trained_models[key] = {
                                "pipeline": pipe,
                                "y_test": y_test,
                                "y_pred": y_pred,
                                "y_proba": y_proba
                            }

                            # Progress allocation across models
                            base, end = 40, 100
                            prog.progress(base + int((end - base) * (i / n)))

                        status.empty()

                        # 4) Persist & show results
                        results_df = pd.DataFrame(results).set_index("model").sort_values("f1", ascending=False)
                        st.session_state.results = results_df
                        st.session_state.models = trained_models
                        st.session_state.feature_schema = schema
                        st.session_state.X_test = X_test

                        st.success("✅ Training complete!")
                        st.balloons()

                        st.subheader("📊 Training Results (Higher F1 is better)")
                        st.dataframe(
                            results_df.style.format({
                                "accuracy": "{:.3f}",
                                "precision": "{:.3f}",
                                "recall": "{:.3f}",
                                "f1": "{:.3f}",
                                "roc_auc": (lambda x: "" if pd.isna(x) else f"{x:.3f}")
                            }).highlight_max(axis=0, color="#E6FFED"),
                            use_container_width=True
                        )

                        if not results_df.empty:
                            top_key = results_df.index[0]
                            st.subheader(f"🏅 Best Model: {top_key.replace('-', ' ').title()}")
                            top_row = results_df.loc[top_key]
                            cA, cB = st.columns(2)
                            with cA:
                                st.metric("Accuracy", f"{top_row['accuracy']*100:,.1f}%")
                                st.metric("Recall", f"{top_row['recall']*100:,.1f}%")
                            with cB:
                                st.metric("Precision", f"{top_row['precision']*100:,.1f}%")
                                st.metric("F1 Score", f"{top_row['f1']*100:,.1f}%")
                            if not pd.isna(top_row.get("roc_auc", np.nan)):
                                st.metric("ROC AUC", f"{top_row['roc_auc']*100:,.1f}%")

                        with st.expander("🔍 Feature Schema"):
                            st.write({
                                "numeric_features": schema.get("numeric_features", []),
                                "categorical_features": schema.get("categorical_features", []),
                            })

                except Exception as e:
                    st.error(f"Error training models: {e}")

# ==================== PAGE 4: MODEL COMPARISON ====================
elif page == "📊 Model Comparison":
    st.header("📊 Model Performance Comparison")
    
    if st.session_state.results is None:
        st.warning("⚠️ Please train models first!")
    else:
        results = st.session_state.results
        models = st.session_state.models
        
        # Metrics comparison
        st.subheader("📈 Performance Metrics")
        
        fig = go.Figure()
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in results.columns:
                fig.add_trace(go.Bar(
                    name=metric.upper(),
                    x=results.index,
                    y=results[metric],
                    text=results[metric].round(3),
                    textposition='auto',
                ))
        
        fig.update_layout(
            barmode='group',
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = results.index[0]
        st.subheader("🏆 Best Model")
        st.success(f"**{best_model}** with F1 Score: {results.loc[best_model, 'f1']:.4f}")
        
        # Confusion matrices
        st.subheader("🔢 Confusion Matrices")
        cols = st.columns(3)
        
        for idx, (name, model_data) in enumerate(models.items()):
            with cols[idx]:
                cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    title=name,
                    color_continuous_scale='Purples'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Deploy to production
        st.subheader("🚀 Deploy Model")
        selected_model = st.selectbox("Select model for production", results.index.tolist())
        
        if st.button("Deploy to Production", type="primary"):
            st.session_state.production_model = models[selected_model]['pipeline']
            st.success(f"✅ {selected_model} deployed to production!")

# ==================== PAGE 5: MAKE PREDICTIONS ====================
elif page == "🔮 Make Predictions":
    st.header("🔮 Predict Customer Churn")
    
    if st.session_state.production_model is None:
        st.warning("⚠️ Please deploy a model first!")
    else:
        pipe = st.session_state.production_model
        schema = st.session_state.feature_schema
        
        st.subheader("📝 Enter Customer Information")
        
        input_data = {}
        
        # Create input fields based on schema
        num_cols = schema['numeric_features']
        cat_cols = schema['categorical_features']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numeric Features**")
            for col in num_cols:
                input_data[col] = st.number_input(f"{col}", value=0.0, key=f"num_{col}")
        
        with col2:
            st.markdown("**Categorical Features**")
            for col in cat_cols:
                # Get unique values from cleaned data if available
                if st.session_state.df_clean is not None and col in st.session_state.df_clean.columns:
                    options = st.session_state.df_clean[col].unique().tolist()
                    input_data[col] = st.selectbox(f"{col}", options, key=f"cat_{col}")
                else:
                    input_data[col] = st.text_input(f"{col}", key=f"text_{col}")
        
        if st.button("🎯 Predict Churn", type="primary"):
            try:
                # Create DataFrame with correct column order
                X = pd.DataFrame([input_data], columns=num_cols + cat_cols)
                
                prediction = pipe.predict(X)[0]
                proba = pipe.predict_proba(X)[0, 1]
                
                st.markdown("---")
                st.subheader("🔮 Prediction Result")
                
                if prediction == 1:
                    st.error("⚠️ **Customer will CHURN**")
                    st.markdown(f"### Churn Probability: {proba*100:.1f}%")
                else:
                    st.success("✅ **Customer will NOT churn**")
                    st.markdown(f"### Retention Probability: {(1-proba)*100:.1f}%")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={'text': "Churn Risk"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if proba > 0.5 else "darkgreen"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "yellow"},
                            {'range': [66, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 App Status")
if st.session_state.df_raw is not None:
    st.sidebar.success("✅ Data Uploaded")
else:
    st.sidebar.info("⏳ Awaiting Data")

if st.session_state.df_clean is not None:
    st.sidebar.success("✅ Data Cleaned")
else:
    st.sidebar.info("⏳ Awaiting Cleaning")

if st.session_state.results is not None:
    st.sidebar.success("✅ Models Trained")
else:
    st.sidebar.info("⏳ Awaiting Training")

if st.session_state.production_model is not None:
    st.sidebar.success("✅ Model Deployed")
else:
    st.sidebar.info("⏳ Awaiting Deployment")
