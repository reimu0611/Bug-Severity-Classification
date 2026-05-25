import streamlit as st
import pickle
import re
import string
import pandas as pd
import numpy as np
import random
import os
import html as _h

# --- Page Configuration ---
st.set_page_config(
    page_title="GitHub Issue Predictor",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INJECTION CUSTOM CSS UNTUK TAMPILAN PREMIUM ---
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Google Sans', sans-serif; }
    .main { background-color: #202225; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible;}

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: #292b2f;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem !important;
        font-weight: 600 !important; 
        padding: 8px 0 !important;
    }
    [data-testid="stSidebar"] hr { border-color: #40444b !important; }

    /* ── HERO ── */
    .hero-box {
        background: linear-gradient(135deg, #033c80 0%, #0054ba 50%, #006aeb 100%);
        border-radius: 24px;
        padding: 48px 40px;
        text-align: center;
        margin-bottom: 32px;
        box-shadow: 0 4px 16px rgba(0, 106, 235, 0.25);
    }
    .hero-title {
        font-family: 'Space Grotesk', serif;
        font-size: 3rem;
        color: white;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero-subtitle { font-size: 1.1rem; color: rgba(255,255,255,0.85); margin-top: 8px; }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.4);
        color: white;
        border-radius: 20px;
        padding: 4px 16px;
        font-size: 0.85rem;
        margin-top: 12px;
    }

    /* ── SECTION TITLE ── */
    .section-title { 
        font-family: 'Google Sans', serif;
        font-size: 1.8rem; 
        color: white; 
        padding-left: 16px;
        margin: 24px 0 20px 0; 
        position: relative;
    }
    .section-title::before {
        content: "";
        position: absolute;
        left: 0;
        top: 10%;
        bottom: 10%;
        width: 4px;
        background-color: #40444b; 
        border-radius: 50px;
    }
            
    .section-desc {
        background: #292b2f;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #40444b;
        color: white;
        line-height: 1.7;
        margin-bottom: 24px;
    }

    /* ── INFO CARDS ── */
    .info-card {
        background: #292b2f;
        border-radius: 14px;
        padding: 20px 22px;
        border: 1px solid #40444b;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
    .info-card h4 { color: white; margin-bottom: 8px; font-size: 1rem; }

    /* ── STEP BADGE ── */          
    .step-badge {
        display: inline-block;
        background: linear-gradient(135deg, #033c80, #0054ba);
        color: white;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        line-height: 36px;
        text-align: center;
        font-weight: 900;
        font-size: 1rem;
        margin-right: 10px;
        vertical-align: middle;
    }

    /* ── BUTTONS ── */ 
    div.stButton > button {
        background: linear-gradient(135deg, red, #40444b);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-family: 'Nunito', sans-serif;
        font-weight: 800;
        font-size: 1rem;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 4px 16px rgba(255, 0, 0, 0.25);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 0, 0, 0.35);
    }
    
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 32px;
        padding-top: 16px;
        border-top: 1px solid #40444b;
    }
    .main .block-container { padding-bottom: 1rem !important; }
</style>  
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # 1. Deklarasi awal semua variabel agar tidak error jika file tidak ada
    data_splits = None
    lr_model = None
    mnb_model = None
    tuned_lr_model = None
    nb_biner_model = None
    svm_biner_model = None
    tfidf_vectorizer = None

    # 2. Setup path - gunakan __file__ untuk mendapatkan lokasi app.py
    # Kemudian naik 1 level ke project root, lalu akses folder data
    if os.path.exists("data/processed"):
        # Ini jalur yang akan dipakai oleh Streamlit Cloud (karena jalan dari root)
        base_path = "data/processed"
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))  # D:\...\web
        project_root = os.path.dirname(app_dir)  # D:\...\Bug-Severity-Classification
        base_path = os.path.join(project_root, 'data', 'processed')

    # 3. Load secara independen satu per satu
    try:
        with open(os.path.join(base_path, 'data_splits_tfidf.pkl'), 'rb') as f:
            data_splits = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading data_splits: {e}")
        pass

    try:
        with open(os.path.join(base_path, 'model_lr_tuned_biner.pkl'), 'rb') as f:
            lr_model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading lr_model: {e}")
        pass

    try:
        with open(os.path.join(base_path, 'model_nb_biner.pkl'), 'rb') as f:
            mnb_model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading mnb_model: {e}")
        pass

    try:
        with open(os.path.join(base_path, 'model_lr_tuned_biner.pkl'), 'rb') as f:
            tuned_lr_model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading tuned_lr_model: {e}")
        pass

    try:
        with open(os.path.join(base_path, 'model_nb_biner.pkl'), 'rb') as f:
            nb_biner_model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading nb_biner_model: {e}")
        pass

    try:
        with open(os.path.join(base_path, 'model_svm_biner.pkl'), 'rb') as f:
            svm_biner_model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading svm_biner_model: {e}")
        pass

    try:
        with open(os.path.join(base_path, 'tfidf_vectorizer.pkl'), 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading tfidf_vectorizer: {e}")
        pass

    return data_splits, lr_model, mnb_model, tuned_lr_model, nb_biner_model, svm_biner_model, tfidf_vectorizer

# Unpacking 7 variabel (HARUS MATCH)
data_splits, lr_model, mnb_model, tuned_lr_model, nb_biner_model, svm_biner_model, tfidf_vectorizer = load_models()

# Label encoder (jika tersedia) untuk memastikan mapping label konsisten
label_encoder = None
if isinstance(data_splits, dict):
    label_encoder = data_splits.get('label_encoder')

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _map_class_to_label(class_value):
    """Map numeric/label class to UI label using label_encoder if available."""
    label = None
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        try:
            if isinstance(class_value, (int, np.integer)):
                label = label_encoder.classes_[int(class_value)]
            else:
                label = str(class_value)
        except Exception:
            label = str(class_value)
    else:
        # C ada sebelum N di alfabet, jadi Critical adalah 0
        label = "Critical" if class_value == 0 else "Non-Critical"

    if str(label).strip().lower() == "critical":
        return "🚨 CRITICAL"
    return "✅ NON-CRITICAL"

def _build_proba_dict(model, proba):
    probas = {}
    classes = getattr(model, "classes_", [])
    for cls, p in zip(classes, proba):
        probas[_map_class_to_label(cls)] = p * 100
    probas.setdefault("✅ NON-CRITICAL", 0.0)
    probas.setdefault("🚨 CRITICAL", 0.0)
    return probas

def convert_to_binary(pred_num):
    """Convert numeric class to label text."""
    return _map_class_to_label(pred_num)

# --- LOAD DATA SAMPLE UNTUK DASHBOARD ---
@st.cache_data
def load_sample_data():
    try:
        # 1. Tentukan base_path secara dinamis terlebih dahulu
        if os.path.exists("data/processed"):
            base_path = "data/processed"
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(app_dir)
            base_path = os.path.join(project_root, 'data', 'processed')
            
        # 2. Baca file CSV di luar blok if-else menggunakan base_path yang sudah siap
        csv_path = os.path.join(base_path, 'github_issues_preprocessed.csv')
        df = pd.read_csv(csv_path, nrows=1000)
        
        return df
    except Exception as e:
        # Mengembalikan DataFrame kosong jika file tidak ditemukan atau error
        return pd.DataFrame()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-family:'Google Sans','Regular 400'; font-size:2.5rem; color:white; font-weight:700;">Buginator</div>
        <div style="font-size:0.85rem; color:rgba(255,255,255,0.7); margin-top:4px;">Github Issue Navigator</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        options=[
            "Home",
            "📊 Data Explorer & EDA",
            "⚙️ Feature Extraction",
            "🏆 Model Evaluation & Comparison"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="padding: 0 10px;">
        <div style="font-size:0.8rem; color:white; font-weight:700; margin-bottom:8px; opacity:0.9;">Group 8</div>
        <div style="font-size:0.7rem; color:rgba(255,255,255,0.6); line-height:1.6;">
            1. ANGELINA JOLIE CANDAYA 2802541644<br>
            2. JOSHUA KEVIN LIEM 2802535572<br>
            3. MAUREEN CALISTA SURJO 2802536392<br>
            4. NICHOLAS HUBERT SOEGIHONO 2802515564<br>
            5. ZAHRA' ZAKIYYAH PRIYONO 2802492554
        </div>
        <div style="margin-top:12px; font-size:0.65rem; color:rgba(255,255,255,0.4); text-align:center; border-top:1px solid rgba(255,255,255,0.1); padding-top:8px;">
            NLP Project<br>
            Binus University
        </div>
    </div>
    """, unsafe_allow_html=True)

def hero(title, subtitle, badge="Natural Language Processing Project"):
    st.markdown(f"""
    <div class="hero-box">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
        <div class="hero-badge">{badge}</div>
    </div>
    """, unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

# ===========================================================================
# PAGE: HOME
# ===========================================================================
if page == "Home":
    hero("Buginator", "Github issue predictor")
    st.markdown("<p></p>", unsafe_allow_html=True)

    st.write("") 
    
    # Info tentang binary classification
    st.info("""
    ℹ️ **Binary Classification:** The model predicts whether a GitHub issue is **🚨 CRITICAL** (serious and urgent) or **✅ NON-CRITICAL** (not serious). It uses 3 algorithms: Logistic Regression, Naive Bayes, & SVM.
    """)
    
    issue_text = st.text_area("Enter Issue Details below:", height=180, placeholder="Example: The application crashes whenever I try to upload a file larger than 5MB...")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col2:
        predict_button = st.button("🔥 Run Prediction")

    if predict_button:
        if not issue_text.strip():
            st.error("⚠️ Text must not be empty!")
        else:
            if tfidf_vectorizer is None:
                st.error("❌ Prediction failed: tfidf_vectorizer.pkl file has not been loaded.")
            else:
                with st.spinner("Sending model to AI..."):
                    try:
                        cleaned = clean_text(issue_text)
                        vec_text = tfidf_vectorizer.transform([cleaned])
                        
                        # Dictionary untuk menyimpan prediksi dari 3 model
                        predictions = {}
                        
                        # 1. Logistic Regression
                        if tuned_lr_model is not None:
                            try:
                                pred_lr_num = tuned_lr_model.predict(vec_text)[0]
                                pred_lr = convert_to_binary(pred_lr_num)
                                proba_lr = tuned_lr_model.predict_proba(vec_text)[0]
                                # BINARY: proba[0] = Non-Critical (class 0), proba[1] = Critical (class 1)
                                # BINARY: proba[0] = Critical (class 0), proba[1] = Non-Critical (class 1)
                                confidence_lr = max(proba_lr) * 100

                                predictions['Logistic Regression'] = {
                                    'prediction': pred_lr,
                                    'pred_num': pred_lr_num,
                                    'confidence': confidence_lr,
                                    'probas': {'🚨 CRITICAL': proba_lr[0]*100, '✅ NON-CRITICAL': proba_lr[1]*100}
                                }
                            except Exception as e:
                                pass
                        
                        # 2. Naive Bayes
                        if nb_biner_model is not None:
                            try:
                                pred_nb_num = nb_biner_model.predict(vec_text)[0]
                                pred_nb = convert_to_binary(pred_nb_num)
                                proba_nb = nb_biner_model.predict_proba(vec_text)[0]
                                # BINARY: proba[0] = Non-Critical (class 0), proba[1] = Critical (class 1)
                                # BINARY: proba[0] = Critical (class 0), proba[1] = Non-Critical (class 1)
                                confidence_nb = max(proba_nb) * 100

                                predictions['Multinomial Naïve Bayes'] = {
                                    'prediction': pred_nb,
                                    'pred_num': pred_nb_num,
                                    'confidence': confidence_nb,
                                    'probas': {'🚨 CRITICAL': proba_nb[0]*100, '✅ NON-CRITICAL': proba_nb[1]*100}
                                }
                            except Exception as e:
                                pass
                        
                        # 3. SVM
                        if svm_biner_model is not None:
                            try:
                                pred_svm_num = svm_biner_model.predict(vec_text)[0]
                                pred_svm = convert_to_binary(pred_svm_num)
                                # Untuk SVM (LinearSVC), gunakan decision function untuk confidence
                                try:
                                    decision = svm_biner_model.decision_function(vec_text)[0]
                                    # Hitung confidence dengan sigmoid function
                                    confidence_svm = (1 / (1 + np.exp(-decision))) * 100
                                    if confidence_svm < 50:
                                        confidence_svm = 100 - confidence_svm
                                except:
                                    confidence_svm = 50.0
                                
                                predictions['Support Vector Machine'] = {
                                    'prediction': pred_svm,
                                    'pred_num': pred_svm_num,
                                    'confidence': confidence_svm,
                                    'probas': None
                                }
                            except Exception as e:
                                pass
                        
                        st.markdown("<hr>", unsafe_allow_html=True)
                        
                        # Tampilkan hasil dari ketiga model
                        if predictions:
                            st.markdown("<h3 style='color:white; text-align:center; margin-bottom:20px;'>📊 Prediction Results from 3 Models</h3>", unsafe_allow_html=True)
                            
                            cols = st.columns(len(predictions))
                            
                            for idx, (model_name, result) in enumerate(predictions.items()):
                                with cols[idx]:
                                    pred = result['prediction']
                                    conf = result['confidence']
                                    
                                    # Tentukan warna & icon berdasarkan prediction (BINARY)
                                    # FIX: Check untuk icon, bukan substring (karena "NON-CRITICAL" mengandung "CRITICAL")
                                    if "🚨" in pred:
                                        color = "#FF0000"
                                        icon = "🚨"
                                        desc = "Immediate attention required - Serious issue!"
                                    else:  # NON-CRITICAL
                                        color = "#00FF00"
                                        icon = "✅"
                                        desc = "Low priority - Not a serious issue"
                                    
                                    # Tambah badge "Best F1" untuk SVM
                                    badge_html = ""
                                    if "Support Vector Machine" in model_name:
                                        badge_html = "<div style='background:#FFD700; color:black; padding:2px 8px; border-radius:8px; font-size:0.7rem; font-weight:bold; display:inline-block; margin-bottom:8px;'>⭐ Best F1</div>"
                                    
                                    # st.markdown(f"""
                                    # <div class="info-card" style="border-left: 6px solid {color}; text-align: center;">
                                    #     <div style="margin-bottom:8px;">{badge_html}</div>
                                    #     <h4 style="margin:0; color:{color}; margin-bottom:8px; font-size:0.9rem;">{model_name}</h4>
                                    #     <div style="font-size:2.5rem; margin:12px 0;">{icon}</div>
                                    #     <h3 style="margin:8px 0; color:white; font-size:1.6rem;">{pred}</h3>
                                    #     <p style="color:#99aab5; font-size:0.85rem; margin:8px 0;">{desc}</p>
                                    #     <div style="background:#202225; padding:12px; border-radius:8px; margin-top:12px; border:2px solid {color};">
                                    #         <div style="font-size:2.2rem; color:{color}; font-weight:bold;">{conf:.1f}%</div>
                                    #         <div style="font-size:0.75rem; color:#99aab5; margin-top:4px;">Confidence</div>
                                    #     </div>
                                    # </div>
                                    # """, unsafe_allow_html=True)
                                    st.markdown(f"""
                                        <div class="info-card" style="border-left: 6px solid {color}; text-align: center;">
                                            <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-bottom: 12px;">
                                                <h4 style="margin: 0; color: {color}; font-size: 1.4rem; font-weight: bold;">{model_name}</h4>
                                                <div>{badge_html}</div>
                                            </div>
                                            <div style="font-size:2.5rem; margin:12px 0;">{icon}</div>
                                            <h3 style="margin:8px 0; color:white; font-size:1.6rem;">{pred}</h3>
                                            <p style="color:#99aab5; font-size:0.85rem; margin:8px 0;">{desc}</p>
                                            <div style="background:#202225; padding:12px; border-radius:8px; margin-top:12px; border:2px solid {color};">
                                                <div style="font-size:2.2rem; color:{color}; font-weight:bold;">{conf:.1f}%</div>
                                                <div style="font-size:0.75rem; color:#99aab5; margin-top:4px;">Confidence</div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Tampilkan detail probabilitas
                            st.markdown("<h4 style='color:white; margin-top:24px;'>📈 Class Probability Details</h4>", unsafe_allow_html=True)
                            
                            for model_name, result in predictions.items():
                                if result['probas'] is not None:
                                    st.markdown(f"**{model_name}:**", unsafe_allow_html=True)
                                    
                                    proba_dict = result['probas']
                                    
                                    col_prob1, col_prob2 = st.columns(2)
                                    
                                    with col_prob1:
                                        st.markdown(f"""
                                        <div class="info-card" style="text-align: center; border-left:4px solid #00FF00;">
                                            <div style="color:#99aab5; font-size:0.85rem; margin-bottom:6px; font-weight:bold;">✅ Non-Critical</div>
                                            <div style="font-size:2rem; color:#00FF00; font-weight:bold;">{proba_dict.get('✅ NON-CRITICAL', 0):.1f}%</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col_prob2:
                                        st.markdown(f"""
                                        <div class="info-card" style="text-align: center; border-left:4px solid #FF0000;">
                                            <div style="color:#99aab5; font-size:0.85rem; margin-bottom:6px; font-weight:bold;">🚨 Critical</div>
                                            <div style="font-size:2rem; color:#FF0000; font-weight:bold;">{proba_dict.get('🚨 CRITICAL', 0):.1f}%</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.markdown("")
                        else:
                            st.error("❌ Prediction failed: No model is available.")
                            
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")

    col1, col2 = st.columns([3,2])
    with col1:
        section("Background")
        st.markdown("""
        <div class="section-desc">
            <p>Software projects that use GitHub generate large volumes of issue reports every day. These reports include bug descriptions, feature requests, and general technical discussions. Each issue is typically assigned one or more labels to help organize and categorize the content.</p>
            <p>In practice, these labels are often inconsistent and unstructured. Many GitHub Issue datasets do not provide explicit severity information such as Critical or Non-critical. Instead, severity must be inferred from existing labels that were created for other purposes. These labels are usually stored as raw multi-label text, where a single issue may contain multiple tags in one field.</p>
            <p>This lack of standardized severity labeling creates a challenge for bug triaging. Developers and maintainers must spend additional time interpreting labels, separating multiple tags, and identifying which issues are Critical and which are Non-critical. As the number of issue reports grows, this process becomes increasingly difficult to manage manually.</p>
            <p>Without a clear structure for severity, Critical issues can be misclassified or overlooked. This leads to delays in fixing important bugs and reduces the efficiency of software maintenance workflows.</p>
            <p>To address this gap, there is a need to analyze existing issue labels and understand their distribution and structure. By examining how labels are used in real datasets, it becomes possible to identify patterns that separate Critical and Non-critical issues for later processing stages.</p>
        </div>
        """, unsafe_allow_html=True)

        section("Dataset we are working with")
        st.markdown("""
        <div class="info-card">
            <h4>sharjeelyunus/github-issues-dataset</h4>
            <div style="font-size:0.85rem;color:#fffff;">
                Sumber: <a href="https://huggingface.co/datasets/sharjeelyunus/github-issues-dataset" target="_blank" style="color:#f7f91a;text-decoration:none;"><b>Hugging Face</b></a><br>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        section("NLP project workflow")
        steps = [
            ("1", "Label Construction", "Extract and map raw GitHub labels to binary severity categories (Critical vs Non-Critical)"),
            ("2", "EDA", "Explore dataset statistics, visualize class distribution, and identify data imbalance issues"),
            ("3", "Preprocessing", "Apply text cleaning: lowercasing, regex cleaning, stopword removal, and lemmatization"),
            ("4", "Feature Extraction", "Convert text documents into TF-IDF numerical vectors for model training"),
            ("5", "Modelling LR", "Train Logistic Regression model with hyperparameter tuning and cross-validation"),
            ("6", "Modelling NB", "Train Multinomial Naive Bayes classifier and evaluate performance metrics"),
            ("7", "Modelling SVM", "Train SVM with best F1-score and compare all three algorithms"),
            ("8", "Evaluation", "Comprehensive evaluation using F1-Score, Precision, Recall, and final model selection"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="info-card" style="padding:14px 18px; margin-bottom:10px;">
                <div style="display:flex;align-items:center;margin-bottom:4px;">
                    <span class="step-badge">{num}</span>
                    <b style="color:#006aeb;"> {title}</b>
                </div>
                <div style="font-size:0.82rem;color:white;padding-left:46px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    section("Project Team")
    st.markdown("""
    <div class="info-card" style="background: linear-gradient(135deg, #202225 0%, #2f3136 100%);">
        <b style="color:#006aeb; font-size:1rem; display:block; margin-bottom:10px;">Group 8</b>
        <table style="width:100%; font-size:0.85rem; color:white; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #40444b;">
                <td style="padding:8px 0; font-weight:600;">ANGELINA JOLIE CANDAYA</td>
                <td style="padding:8px 0; text-align:right; color:white;">2802541644</td>
            </tr>
            <tr style="border-bottom:1px solid #40444b;">
                <td style="padding:8px 0; font-weight:600;">JOSHUA KEVIN LIEM</td>
                <td style="padding:8px 0; text-align:right; color:white;">2802535572</td>
            </tr>
            <tr>
                <td style="padding:8px 0; font-weight:600;">MAUREEN CALISTA SURJO</td>
                <td style="padding:8px 0; text-align:right; color:white;">2802536392</td>
            </tr>
            <tr>
                <td style="padding:8px 0; font-weight:600;">NICHOLAS HUBERT SOEGIHONO</td>
                <td style="padding:8px 0; text-align:right; color:white;">2802515564</td>
            </tr>
            <tr>
                <td style="padding:8px 0; font-weight:600;">ZAHRA' ZAKIYYAH PRIYONO</td>
                <td style="padding:8px 0; text-align:right; color:white;">2802492554</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE: DATA EXPLORER & EDA
# ===========================================================================
elif page == "📊 Data Explorer & EDA":
    hero("Data Explorer & EDA", "Analysis of dataset characteristics, class imbalance, and text transformation processes", "Pipeline Stages 1 & 2")
    
    section("Overview Dataset")
    df_sample = load_sample_data()
    
    if df_sample.empty:
        st.error("❌ File github_issues_preprocessed.csv not found in folder data/processed/")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="info-card" style="text-align: center; border-top: 4px solid #006aeb;">
                <h4 style="color: #99aab5; margin: 0; font-size:14px;">Total Clean Dataset</h4>
                <h2 style="color: white; margin: 5px 0;">114,065 <span style="font-size:14px; color:#99aab5;">Rows</span></h2>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("""
            <div class="info-card" style="text-align: center; border-top: 4px solid #006aeb;">
                <h4 style="color: #99aab5; margin: 0; font-size:14px;">Classification Target</h4>
                <h2 style="color: white; margin: 5px 0;">2 <span style="font-size:14px; color:#99aab5;">Severity Class</span></h2>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown("""
            <div class="info-card" style="text-align: center; border-top: 4px solid #006aeb;">
                <h4 style="color: #99aab5; margin: 0; font-size:14px;">Maximum Vocabulary Features</h4>
                <h2 style="color: white; margin: 5px 0;">50,000 <span style="font-size:14px; color:#99aab5;">Important Words</span></h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<p style='color:#99aab5; margin-bottom: 5px;'>Representative sample of processed data (first 1,000 rows):</p>", unsafe_allow_html=True)
        df_sample_view = df_sample.copy()
        df_sample_view['severity'] = df_sample_view['severity'].replace({
            'Major': 'Non-Critical',
            'Minor': 'Non-Critical'
        })
        st.dataframe(df_sample_view, use_container_width=True, height=300)
        
        st.markdown("---")

        section("Exploratory Data Analysis")
        col_chart_text, col_chart_visual = st.columns([2, 3])
        
        with col_chart_text:
            st.markdown("""
            <div class="section-desc" style="height: 100%;">
                <h4 style="margin-top:0; color:#FFD700;">🚨 Class Imbalance Problem</h4>
                Based on the target distribution visualization beside it, this dataset clearly suffers from a severe <b>Class Imbalance</b> problem.
                <br><br>
                The number of data samples labeled as <code>Critical</code> is significantly smaller compared to the <code>Non-Critical</code> class.
                <br><br>
                <b>Practical Impact:</b> The model tends to become better at predicting the Non-Critical class because it has far more training examples. Therefore, the final model performance should not be evaluated using Accuracy alone, but should instead rely on the <b>F1-Macro Average</b> score.
            </div>
            """, unsafe_allow_html=True)
            
        with col_chart_visual:
            st.markdown("""
            <div class='info-card' style='margin-bottom:0;'>
                <h4 style='text-align:center; color:white; font-size:0.95rem;'>Target Severity Frequency Distribution</h4>
            </div>
            """, unsafe_allow_html=True)
            severity_binary = df_sample['severity'].replace({
                'Major': 'Non-Critical',
                'Minor': 'Non-Critical'
            })
            dist_data = severity_binary.value_counts().reindex(['Non-Critical', 'Critical']).fillna(0)
            st.bar_chart(dist_data, color="#006aeb")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p style='color:#99aab5; margin-bottom: 5px;'>Statistics of word length distribution in the report document text:</p>", unsafe_allow_html=True)
        
        c_stat1, c_stat2, c_stat3 = st.columns(3)
        with c_stat1:
            st.markdown("""
            <div class="info-card" style="text-align: center;">
                <div style="font-size:2.2rem; color:#006aeb; font-weight:900;">1</div>
                <div style="color:#99aab5; font-size:0.85rem; font-weight:600; margin-top:4px;">Words (Shortest Text)</div>
            </div>
            """, unsafe_allow_html=True)
        with c_stat2:
            st.markdown("""
            <div class="info-card" style="text-align: center;">
                <div style="font-size:2.2rem; color:#006aeb; font-weight:900;">~108</div>
                <div style="color:#99aab5; font-size:0.85rem; font-weight:600; margin-top:4px;">Words (Average per Text)</div>
            </div>
            """, unsafe_allow_html=True)
        with c_stat3:
            st.markdown("""
            <div class="info-card" style="text-align: center; border-left: 4px solid red;">
                <div style="font-size:2.2rem; color:red; font-weight:900;">12,000+</div>
                <div style="color:#99aab5; font-size:0.85rem; font-weight:600; margin-top:4px;">Words (Longest Outlier)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        section("Text Preprocessing Pipeline")
        st.markdown("""
        <div class="section-desc" style="display: flex; justify-content: space-between; text-align: center; padding: 15px 20px;">
            <div><span class="step-badge" style="width:24px; height:24px; line-height:24px; font-size:0.8rem;">1</span> <b>Lowercasing</b></div>
            <div><span class="step-badge" style="width:24px; height:24px; line-height:24px; font-size:0.8rem;">2</span> <b>Regex Cleaning</b></div>
            <div><span class="step-badge" style="width:24px; height:24px; line-height:24px; font-size:0.8rem;">3</span> <b>Stopwords Removal</b></div>
            <div><span class="step-badge" style="width:24px; height:24px; line-height:24px; font-size:0.8rem;">4</span> <b>Lemmatization</b></div>
        </div>
        """, unsafe_allow_html=True)

        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("""
            <div class="info-card" style="border-top: 4px solid red; height: 100%;">
                <h4 style="color:red; font-weight:700;">🔴 Raw Text Input (Before)</h4>
                <p style="color:#99aab5; font-size:0.8rem; margin-top:-5px;">Original document text from the GitHub server</p>
                <div style="background:#202225; padding:15px; border-radius:8px; color:#eee; font-family:monospace; font-size:0.88rem; line-height:1.6; border: 1px solid #40444b;">
                    [Feature]: Invalid header error should show what the invalid header is.<br><br>
                    When passing an invalid header to fetch() it just throws a generic TypeError: Invalid name error... It would be very helpful if it included the name of the invalid header!
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_after:
            st.markdown("""
            <div class="info-card" style="border-top: 4px solid #006aeb; height: 100%;">
                <h4 style="color:#006aeb; font-weight:700;">🔵 Cleaned Tokens (After)</h4>
                <p style="color:#99aab5; font-size:0.8rem; margin-top:-5px;">Corpus normalization results using NLTK</p>
                <div style="background:#202225; padding:15px; border-radius:8px; color:#eee; font-family:monospace; font-size:0.88rem; line-height:1.6; border: 1px solid #40444b;">
                    feature invalid header error show invalid header <br><br>
                    passing invalid header fetch throw generic typeerror invalid name error would helpful included name invalid header
                </div>
            </div>
            """, unsafe_allow_html=True)

# ===========================================================================
# PAGE: FEATURE ENGINEERING & BASELINE
# ===========================================================================
elif page == "⚙️ Feature Extraction":
    hero("Feature Engineering & Baseline", "Scientific experiments on text representation and dataset splitting", "Pipeline Stages 3 & 7")
    
    section("Data Splitting")
    st.markdown("<p style='color:#99aab5;'>Dataset split proportions for the training, validation, and final testing processes:</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-card" style="text-align: center; border-bottom: 4px solid #006aeb;">
            <h4 style="color: #99aab5; margin: 0; font-size:14px;">Training Data</h4>
            <h2 style="color: white; margin: 5px 0;">79,845 <span style="font-size:14px; color:#99aab5;">Rows</span></h2>
            <div style="font-size:0.8rem; color:#006aeb; font-weight:bold;">~70% Dataset</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="info-card" style="text-align: center; border-bottom: 4px solid #FFD700;">
            <h4 style="color: #99aab5; margin: 0; font-size:14px;">Validation Data</h4>
            <h2 style="color: white; margin: 5px 0;">17,110 <span style="font-size:14px; color:#99aab5;">Rows</span></h2>
            <div style="font-size:0.8rem; color:#FFD700; font-weight:bold;">~15% Dataset</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="info-card" style="text-align: center; border-bottom: 4px solid red;">
            <h4 style="color: #99aab5; margin: 0; font-size:14px;">Test Data</h4>
            <h2 style="color: white; margin: 5px 0;">17,110 <span style="font-size:14px; color:#99aab5;">Rows</span></h2>
            <div style="font-size:0.8rem; color:red; font-weight:bold;">~15% Dataset</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    section("Text Representation Experiment (Baseline)")
    st.markdown("""
    <div class="section-desc">
        Computers cannot directly understand raw text. Therefore, we need to conduct experiments to compare two feature extraction methods: <b>TF-IDF Vectorizer</b> and <b>CountVectorizer (as the baseline)</b>. Both are limited to a maximum of <b>50,000 of the most relevant words</b> to maintain memory efficiency.
    </div>
    """, unsafe_allow_html=True)

    col_table, col_text = st.columns([4, 5])
    
    with col_table:
        st.markdown("<h4 style='color:white; font-size:1rem; margin-bottom:10px;'>Accuracy Comparison Table (Logistic Regression)</h4>", unsafe_allow_html=True)
        baseline_data = {
            "Representation Method": ["CountVectorizer", "TF-IDF Vectorizer"],
            "Accuracy": ["80.50%", "80.36%"],
            "Principle": ["Calculates Pure Frequency", "Word Importance Weight"]
        }
        st.dataframe(pd.DataFrame(baseline_data), hide_index=True, use_container_width=True)
        
    with col_text:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #FFD700; height: 100%;">
            <h4 style="color:#FFD700; margin-top:0;">💡 Experiment Conclusion</h4>
            The results on the <i>Test</i> data show that <b>TF-IDF</b> (80.36%) and <b>CountVectorizer</b> (80.50%) produce nearly identical accuracy levels.
            <br><br>
            This provides an important <i>insight</i>: <b>the main problem (bottleneck) in this project does not lie in the text representation method</b>, but rather in the <i>Class Imbalance</i> issue (which we observed during EDA) and the ability of the Machine Learning algorithm itself to separate minority class patterns.
        </div>
        """, unsafe_allow_html=True)

# ===========================================================================
# PAGE: MODEL EVALUATION & COMPARISON
# ===========================================================================
elif page == "🏆 Model Evaluation & Comparison":
    hero("Model Evaluation & Comparison", "In-depth evaluation, algorithm comparison, and final prediction performance", "Stages 4, 5, 6, & 7")
    
    section("Head-to-Head: LR vs Multinomial NB vs Support Vector Machine")
    st.markdown("""
    <div class="section-desc">
        Since our dataset suffers from a <b>Class Imbalance</b> problem, where the Critical class is highly underrepresented, the Accuracy metric alone can be misleading. Therefore, we compare the performance of <b>Logistic Regression (LR)</b> and <b>Multinomial Naïve Bayes (MNB)</b> with the main focus on the <b>F1-Score for the Critical class</b> metric.
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="info-card" style="border-top: 4px solid #006aeb; height: 100%;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="color:#006aeb; margin:0;">Logistic Regression</h3>
            </div>
            <hr style="border-color:#40444b; margin:10px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#99aab5; font-size:1.1rem;">Overall Accuracy:</span>
                <span style="color:white; font-size:1.2rem; font-weight:bold;">85.99%</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; background:#202225; padding:10px; border-radius:8px; border:1px solid #006aeb;">
                <span style="color:#99aab5; font-size:1.1rem; font-weight:bold;">F1-Score Critical:</span>
                <span style="color:#006aeb; font-size:1.8rem; font-weight:900;">0.87</span>
            </div>
            <p style="color:#99aab5; font-size:0.85rem; margin-top:10px;">This algorithm is highly robust in learning minority class patterns and handling the <i>sparse</i> text representation produced by TF-IDF.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card" style="border-top: 4px solid #FFD700; height: 100%;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="color:#FFD700; margin:0;">Multinomial Naïve Bayes</h3>
            </div>
            <hr style="border-color:#40444b; margin:10px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#99aab5; font-size:1.1rem;">Overall Accuracy:</span>
                <span style="color:white; font-size:1.2rem; font-weight:bold;">78.24%</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; background:#202225; padding:10px; border-radius:8px; border:1px solid #FFD700;">
                <span style="color:#99aab5; font-size:1.1rem; font-weight:bold;">F1-Score Critical:</span>
                <span style="color:#FFD700; font-size:1.8rem; font-weight:900;">0.79</span>
            </div>
            <p style="color:#99aab5; font-size:0.85rem; margin-top:10px;">Highly sensitive to data imbalance. MNB more frequently misclassifies the Critical class as the majority class (Non-Critical).</p>
        </div>
        """, unsafe_allow_html=True)
    
    
    with c3:
        st.markdown("""
        <div class="info-card" style="border-top: 4px solid red; height: 100%;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="color:red; margin:0;">Support Vector Machine</h3>
                <span style="background:red; color:white; padding:4px 12px; border-radius:12px; font-size:0.8rem; font-weight:bold;">WINNER</span>
            </div>
            <hr style="border-color:#40444b; margin:10px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span style="color:#99aab5; font-size:1.1rem;">Overall Accuracy:</span>
                <span style="color:white; font-size:1.2rem; font-weight:bold;">93.74%</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; background:#202225; padding:10px; border-radius:8px; border:1px solid red;">
                <span style="color:#99aab5; font-size:1.1rem; font-weight:bold;">F1-Score Critical:</span>
                <span style="color:red; font-size:1.8rem; font-weight:900;">0.94</span>
            </div>
            <p style="color:#99aab5; font-size:0.85rem; margin-top:10px;">The decision boundary is robust, stable, and generalizes well to both classes rather than favoring one over the other.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    section("Performance Visualization (Support Vector Machine)")
    
    col_lc, col_cm = st.columns(2)
    
    with col_lc:
        st.markdown("""
        <div class="info-card" style="height: 100%;">
            <h4 style="color:#FFD700; text-align:center;">📈 Learning Curve</h4>
        """, unsafe_allow_html=True)
        
        try:
            st.image("../reports/learning_curve_comparison.png", use_container_width=True)
            st.markdown("""
            <div style="color:#99aab5; font-size:0.85rem; margin-top:10px; line-height:1.5;">
                <b>Interpretation:</b> This graph proves that the SVM model <b>does not experience overfitting</b>. The <i>Training Accuracy</i> and <i>Validation Accuracy</i> curves remain stable, close to each other, and converge as the amount of training data increases.
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.error("⚠️ File 'learning_curve_comparison.png' not found in the directory.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cm:
        st.markdown("""
        <div class="info-card" style="height: 100%;">
            <h4 style="color:#FFD700; text-align:center;">🧩 Confusion Matrix</h4>
        """, unsafe_allow_html=True)
        
        try:
            st.image("../reports/confusion_matrix_comparison.png", use_container_width=True)
            st.markdown("""
            <div style="color:#99aab5; font-size:0.85rem; margin-top:10px; line-height:1.5;">
                <b>Interpretation:</b> This matrix maps both correct predictions (main diagonal) and misclassifications. The most common classification error occurs when the model predicts a <code>Critical</code> report as <code>Non-Critical</code>, which linguistically often share similar words (<i>overlapping keywords</i>).
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.error("⚠️ File 'confusion_matrix_comparison.png' not found in the directory.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    col_leftc, col_centerm, col_rightm = st.columns([0.5,1,0.5])

    with col_centerm:
        st.markdown("""
        <div class="info-card" style="height: 100%;">
            <h4 style="color:#FFD700; text-align:center;">📊 F1 Score Comparison</h4>
        """, unsafe_allow_html=True)
        
        try:
            st.image("../reports/f1_score_comparison_grouped.png", use_container_width=True)
            st.markdown("""
            <div style="color:#99aab5; font-size:0.85rem; margin-top:10px; line-height:1.5;">
                <b>Interpretation:</b> SVM is the best model because it achieves the highest and most balanced F1-scores (≥ 0.930) for both classes. The consistent performance hierarchy across categories (<i>SVM > Logistic Regression > Naive Bayes</i>) proves that SVM creates the most accurate decision boundary for your data with minimal class bias.
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.error("⚠️ File 'f1_score_comparison_grouped.png' not found in the directory.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    section("Final Classification Report (Support Vector Machine)")
    st.markdown("<p style='color:#99aab5;'>Detailed classification evaluation report for the best Support Vector Machine model on the Test Data (17,110 reports):</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#99aab5; font-size:0.8rem;'>Note: After converting to a binary class, the metric values need to be recalculated based on the latest evaluation results.</p>", unsafe_allow_html=True)
    
    report_data = {
        "Class / Label": ["Critical", "Non-Critical", "accuracy", "macro avg", "weighted avg"],
        "Precision": ["0.99", "0.87", "-", "0.93", "0.94"],
        "Recall": ["0.90", "0.99", "-", "0.94", "0.93"],
        "F1-Score": ["0.94", "0.93", "0.93", "0.93", "0.94"],
        "Support": ["13294", "9477", "22771", "22771", "22771"]
    }
    
    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report, use_container_width=True)
