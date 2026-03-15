import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
import math
import umap
import plotly.express as px
import time
import traceback

# --- 1. Page Configuration & Advanced Custom CSS ---
st.set_page_config(
    page_title="DeepSea-AI Classifier",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a stunning, modern look with animations
st.markdown("""
<style>
    .stApp { background: linear-gradient(-45deg, #0c101c, #121827, #003049, #0c101c); background-size: 400% 400%; animation: gradient 15s ease infinite; color: #e0e0e0; }
    @keyframes gradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    .st-emotion-cache-16txtl3 { background-color: rgba(18, 24, 39, 0.8); backdrop-filter: blur(5px); border-right: 1px solid #2d3b53; }
    .st-emotion-cache-z5fcl4, .metric-card { border: 1px solid #2d3b53; border-radius: 10px; padding: 20px !important; background-color: rgba(18, 24, 39, 0.8); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4); transition: all 0.3s ease-in-out; }
    .st-emotion-cache-z5fcl4:hover, .metric-card:hover { box-shadow: 0 0 20px rgba(0, 180, 216, 0.6); border-color: #00b4d8; }
    .stButton>button { background-color: #00b4d8; color: white; border-radius: 8px; border: none; padding: 10px 20px; transition: all 0.3s ease; box-shadow: 0 0 15px rgba(0, 180, 216, 0.4); }
    .stButton>button:hover { background-color: #0077b6; box-shadow: 0 0 25px rgba(0, 180, 216, 0.7); transform: scale(1.02); }
    h1, h2, h3 { color: #ade8f4; text-shadow: 0 0 5px #00b4d8; }
    .metric-card { padding: 15px; text-align: center; } .metric-card-label { font-size: 1.1em; color: #94a3b8; } .metric-card-value { font-size: 2em; font-weight: bold; color: #ffffff; } .metric-card-delta { font-size: 1.2em; color: #00b4d8; }
    .loading-screen { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(12, 16, 28, 0.9); display: flex; flex-direction: column; justify-content: center; align-items: center; z-index: 9999; }
    .loading-spinner { width: 50px; height: 50px; border: 5px solid rgba(0, 180, 216, 0.3); border-radius: 50%; border-top-color: #00b4d8; animation: spin 1s ease-in-out infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading-text { margin-top: 20px; color: #ade8f4; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)

# --- 2. Configuration & Constants ---
MODEL_PATH = 'best_taxonomy_classifier.pth'
LABEL_COLUMNS = ['phylum', 'class', 'order', 'family', 'genus']
FIXED_SEQUENCE_LENGTH = 1000
CONFIDENCE_THRESHOLD = 0.70
LARGE_FILE_WARNING_MB = 50 

# --- 3. Model Architecture ---
class DynamicTaxonomyCNN(nn.Module):
    def __init__(self, params, num_classes_dict):
        super(DynamicTaxonomyCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=params['filters_l1'], kernel_size=8, padding=3),
            nn.BatchNorm1d(params['filters_l1']), nn.ReLU(), nn.MaxPool1d(kernel_size=4), nn.Dropout(params['dropout_l1']),
            nn.Conv1d(in_channels=params['filters_l1'], out_channels=params['filters_l2'], kernel_size=8, padding=3),
            nn.BatchNorm1d(params['filters_l2']), nn.ReLU(), nn.MaxPool1d(kernel_size=4), nn.Dropout(params['dropout_l2'])
        )
        self.flatten = nn.Flatten()
        
        flattened_size = params['filters_l2'] * 62
        
        self.dense_block = nn.Sequential(
            nn.Linear(in_features=flattened_size, out_features=params['dense_units']), 
            nn.ReLU(), 
            nn.Dropout(params['dropout_dense'])
        )
        
        self.phylum_head = nn.Linear(params['dense_units'], num_classes_dict['phylum'])
        self.class_head = nn.Linear(params['dense_units'], num_classes_dict['class'])
        self.order_head = nn.Linear(params['dense_units'], num_classes_dict['order'])
        self.family_head = nn.Linear(params['dense_units'], num_classes_dict['family'])
        self.genus_head = nn.Linear(params['dense_units'], num_classes_dict['genus'])

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x) 
        embedding = self.dense_block(x)
        return {
            'embedding': embedding, 
            'phylum': self.phylum_head(embedding), 
            'class': self.class_head(embedding),
            'order': self.order_head(embedding), 
            'family': self.family_head(embedding), 
            'genus': self.genus_head(embedding)
        }

# --- 4. Caching and Loading Functions ---
@st.cache_resource
def load_model_and_dependencies():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'hyperparameters' in checkpoint:
            best_params = checkpoint['hyperparameters']
            num_classes = checkpoint['num_classes']
            test_accuracies = checkpoint.get('test_accuracies', {})
            model_state_dict = checkpoint['model_state_dict']
        else:
            st.error(f"**Fatal Error: Incomplete Model File**")
            return None, None, None, None, None
            
        inverse_mappings = {col: {v: k for k, v in json.load(open(f"{col}_mapping.json", 'r')).items()} for col in LABEL_COLUMNS}
        
        model = DynamicTaxonomyCNN(best_params, num_classes).to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        return model, inverse_mappings, test_accuracies, num_classes, device
    except Exception as e:
        st.error(f"Fatal Error loading assets: {e}")
        return None, None, None, None, None

# --- 5. Helper Functions ---
def parse_fasta(file_content_string):
    sequences = {}
    current_header = ""
    for line in file_content_string.splitlines():
        if line.startswith(">"): 
            current_header = line[1:].strip()
            sequences[current_header] = ""
        else:
            if current_header: sequences[current_header] += line.strip().upper()
    return sequences

def preprocess_sequences(sequences_dict):
    nuc_map = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,0], 'N': [0,0,0,0]}
    encoded = np.zeros((len(sequences_dict), 4, FIXED_SEQUENCE_LENGTH), dtype=np.uint8)
    for i, seq in enumerate(sequences_dict.values()):
        seq_str = str(seq).upper()
        for j, nuc in enumerate(seq_str[:FIXED_SEQUENCE_LENGTH]):
            if nuc in nuc_map: 
                encoded[i, :, j] = nuc_map[nuc]
    return torch.tensor(encoded, dtype=torch.float32)

def predict_batch(model, sequence_tensor, device, batch_size=128):
    # This try/except wraps the ENTIRE logic and safely returns the error as text
    try:
        dataset = TensorDataset(sequence_tensor)
        loader = DataLoader(dataset, batch_size=batch_size)
        all_predictions = []
        all_embeddings = []
        
        with torch.no_grad():
            for (batch_sequences,) in loader:
                batch_sequences = batch_sequences.to(device)
                outputs = model(batch_sequences)
                
                all_embeddings.append(outputs['embedding'].detach().cpu().numpy())
                
                for i in range(batch_sequences.size(0)):
                    pred_row = {}
                    max_confidence = 0
                    for rank in LABEL_COLUMNS:
                        probs = torch.softmax(outputs[rank][i], dim=0)
                        confidence, pred_idx = torch.max(probs, dim=0)
                        pred_row[rank] = {'index': pred_idx.item(), 'confidence': confidence.item()}
                        if confidence.item() > max_confidence: 
                            max_confidence = confidence.item()
                    
                    pred_row['max_confidence'] = max_confidence
                    pred_row['status'] = "Known" if max_confidence >= CONFIDENCE_THRESHOLD else "Potentially Novel"
                    all_predictions.append(pred_row)
                    
        return all_predictions, np.vstack(all_embeddings)
    except Exception as e:
        # Instead of crashing, we return the error back to the app page
        error_trace = traceback.format_exc()
        return [{"error_caught": True, "trace": error_trace}], None

# --- 6. UI Page Functions ---
def page_live_classifier(model, inverse_mappings, device):
    st.header("🔬 Live Single Sequence Classifier")
    with st.container(border=True):
        default_seq = "GATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACAGATTACA"
        if 'single_seq_input' not in st.session_state: st.session_state.single_seq_input = default_seq
        sequence_input = st.text_area("Enter a DNA sequence below:", key="single_seq_input", height=150)
        
        if st.button("Classify Sequence", key="single_seq_button", use_container_width=True):
            if sequence_input.strip():
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    st.markdown("""
                    <div class="loading-screen">
                        <div class="loading-spinner"></div>
                        <div class="loading-text">Analyzing sequence...</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                predictions, embeddings = predict_batch(model, preprocess_sequences({"seq": sequence_input}), device)
                loading_placeholder.empty()
                
                # If embeddings is None, our error trapper caught a crash!
                if embeddings is None:
                    st.error("🚨 PyTorch Crash Captured! Streamlit can't hide it this time.")
                    st.code(predictions[0]["trace"])
                    st.info("Please copy and paste this stack trace back into our chat.")
                    return
                
                st.success("Classification Complete!")
                results = [[r.capitalize(), inverse_mappings[r].get(predictions[0][r]['index'],'Err'), predictions[0][r]['confidence']] for r in LABEL_COLUMNS]
                df_results = pd.DataFrame(results, columns=["Rank", "Prediction", "Confidence"])
                
                st.subheader("Predicted Lineage")
                try:
                    st.table(df_results.style.format({'Confidence': '{:.2%}'}).background_gradient(cmap='Blues', subset=['Confidence']))
                except ImportError:
                    st.table(df_results.style.format({'Confidence': '{:.2%}'}))

                st.subheader("Prediction Confidence Breakdown")
                for _, row in df_results.iterrows():
                    st.write(f"**{row['Rank']}: {row['Prediction']}**")
                    st.progress(row['Confidence'])
                    if row['Confidence'] < CONFIDENCE_THRESHOLD: 
                        st.warning("Low confidence suggests a novel or poorly represented taxon.")
            else: 
                st.warning("Please enter a DNA sequence.")

def page_biodiversity_dashboard(model, inverse_mappings, device):
    st.header("📊 Batch File Biodiversity Dashboard")
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload a FASTA file of eDNA reads", type=["fasta", "fa", "txt"])
        if uploaded_file:
            file_content = uploaded_file.read().decode("utf-8")
            sequences = parse_fasta(file_content)
            
            if not sequences: 
                st.error("No valid sequences found in the uploaded file.")
                return
            
            max_seq = st.slider("Select number of sequences to analyze:", 1, len(sequences), min(500, len(sequences)))
            sequences_to_process = dict(list(sequences.items())[:max_seq])

            if st.button(f"Analyze {max_seq} Sequences", key="batch_button", use_container_width=True):
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    st.markdown("""
                    <div class="loading-screen">
                        <div class="loading-spinner"></div>
                        <div class="loading-text">Processing sequences...</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Processing... {i+1}%")
                    time.sleep(0.01) 
                
                with st.spinner(f"Classifying {len(sequences_to_process)} sequences on {device.type.upper()}..."):
                    predictions, embeddings = predict_batch(model, preprocess_sequences(sequences_to_process), device)
                
                loading_placeholder.empty()
                progress_bar.empty()
                status_text.empty()
                
                if embeddings is None:
                    st.error("🚨 PyTorch Crash Captured!")
                    st.code(predictions[0]["trace"])
                    return
                
                st.success("Batch analysis complete!")
                
                headers = list(sequences_to_process.keys())
                results_list = [{'header': h, 'status': p['status'], **{f'{r}_pred': inverse_mappings[r].get(p[r]['index'], 'N/A') for r in LABEL_COLUMNS}} for h, p in zip(headers, predictions)]
                results_df = pd.DataFrame(results_list)

                st.subheader("Biodiversity Overview")
                st.dataframe(results_df)

def page_model_details(test_accuracies, num_classes):
    st.header("⚙️ About the AI Model")
    col1, col2 = st.columns([1,1])
    
    with col1:
        with st.container(border=True):
            st.subheader("Model Architecture")
            st.write("A **1D Convolutional Neural Network (CNN)** with a multi-head output, built in PyTorch.")
            st.image("cnn_architecture.png", caption="Detailed Diagram of the 1D CNN Architecture")
            
    with col2:
        with st.container(border=True):
            st.subheader("Model Performance")
            st.write("Final accuracy scores evaluated on a held-out test set.")
            if test_accuracies:
                acc_df = pd.DataFrame(test_accuracies.items(), columns=['Taxonomic Rank', 'Accuracy'])
                acc_df['Taxonomic Rank'] = acc_df['Taxonomic Rank'].str.capitalize()
                st.table(acc_df.style.format({'Accuracy': '{:.2%}'}))
                
    with st.container(border=True):
        st.subheader("Training Data Overview")
        if num_classes:
            class_counts = pd.DataFrame(num_classes.items(), columns=['Rank', 'Unique Categories'])
            class_counts['Rank'] = class_counts['Rank'].str.capitalize()
            fig = px.bar(class_counts, x='Rank', y='Unique Categories', title='Classes per Rank in Training Data', text_auto=True)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

# --- 7. Main App Logic ---
st.sidebar.title("🌊 DeepSea-AI")
page = st.sidebar.radio("Navigation", ["🔬 Live Classifier", "📊 Biodiversity Dashboard", "⚙️ About the Model"])
st.sidebar.markdown("---")
st.sidebar.info("This application is the final deliverable of the AI-driven pipeline for deep-sea eDNA analysis.")

with st.spinner("Initializing AI model... This may take a moment on first run."):
    model, inverse_mappings, test_accuracies, num_classes_dict, device = load_model_and_dependencies()

if model:
    st.sidebar.markdown("---")
    st.sidebar.success(f"Model loaded successfully on **{device.type.upper()}**")
    
    if page == "🔬 Live Classifier": 
        page_live_classifier(model, inverse_mappings, device)
    elif page == "📊 Biodiversity Dashboard": 
        page_biodiversity_dashboard(model, inverse_mappings, device)
    elif page == "⚙️ About the Model": 
        page_model_details(test_accuracies, num_classes_dict)
else:
    st.error("Application could not start. Please check file requirements.")
    st.stop()