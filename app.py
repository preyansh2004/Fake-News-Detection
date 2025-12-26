import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import plotly.graph_objects as go
import time
from datetime import datetime


# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="TruthLens - AI Fake News Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="auto"
)


# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Header */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .hero-description {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.8;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Input Section */
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid #e0e7ff;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Results Section */
    .result-container {
        margin: 2rem 0;
    }
    
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-2px);
    }
    
    .real-news {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
    }
    
    .fake-news {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);
    }
    
    .prediction-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .prediction-text {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .confidence-text {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Progress Bar */
    .progress-container {
        margin: 1rem 0;
        background-color: rgba(255,255,255,0.2);
        border-radius: 10px;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 8px;
        background: rgba(255,255,255,0.8);
        border-radius: 10px;
        transition: width 1s ease;
    }
    
    /* Stats Cards */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.2rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    /* Info Boxes */
    .info-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #1e293b;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .info-box ul {
        list-style: none;
        padding: 0;
    }
    
    .info-box li {
        padding: 0.3rem 0;
        color: #475569;
        position: relative;
        padding-left: 1.5rem;
    }
    
    .info-box li:before {
        content: "▶";
        color: #667eea;
        position: absolute;
        left: 0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        margin: 1rem 0;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-subtitle {
            font-size: 1.1rem;
        }
        
        .input-section {
            padding: 1rem;
            margin: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)


# Download NLTK data (cached for performance)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)


# Load the model (cached for performance)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('fake_news_model.joblib')
        return model
    except FileNotFoundError:
        st.error("🚨 Model file 'fake_news_model.joblib' not found. Please ensure the model file is in the same directory as this script.")
        return None


# Preprocessing function (same as training)
@st.cache_data
def enhanced_preprocessing(content):
    port_stem = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Original stemming
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    words = stemmed_content.split()
    stemmed_words = [port_stem.stem(word) for word in words if word not in stop_words]
    processed_text = ' '.join(stemmed_words)
    
    # Linguistic features
    features = []
    features.append(f"EXCL_{min(content.count('!'), 10)}")
    features.append(f"QUEST_{min(content.count('?'), 10)}")
    
    # ALL CAPS detection
    all_caps = sum(1 for word in words if word.isupper() and len(word) > 1)
    features.append(f"CAPS_{min(all_caps, 5)}")
    
    # Quote detection
    quotes = content.count('"') + content.count("'")
    features.append(f"QUOTES_{min(quotes, 10)}")
    
    return processed_text + ' ' + ' '.join(features)


# Function to get prediction confidence
def get_prediction_confidence(model, text):
    decision_scores = model.decision_function([text])[0]
    confidence = (1 / (1 + np.exp(-abs(decision_scores)))) * 100
    return confidence


# Function to analyze text features
def analyze_text_features(text):
    words = text.split()
    sentences = text.split('.')
    
    features = {
        'Word Count': len(words),
        'Sentence Count': len([s for s in sentences if s.strip()]),
        'Character Count': len(text),
        'Exclamation Marks': text.count('!'),
        'Question Marks': text.count('?'),
        'Quotation Marks': text.count('"') + text.count("'"),
        'All Caps Words': sum(1 for word in words if word.isupper() and len(word) > 1),
        'Average Word Length': round(np.mean([len(word) for word in words]) if words else 0, 1),
        'Average Sentence Length': round(len(words) / len([s for s in sentences if s.strip()]) if sentences else 0, 1)
    }
    return features


def create_feature_chart(features):
    # Create a more visually appealing chart
    fig = go.Figure()
    
    # Separate numeric features for better visualization
    chart_features = {k: v for k, v in features.items() if k not in ['Average Word Length', 'Average Sentence Length']}
    
    fig.add_trace(go.Bar(
        x=list(chart_features.values()),
        y=list(chart_features.keys()),
        orientation='h',
        marker=dict(
            color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b'],
            line=dict(color='rgba(255,255,255,0.8)', width=1)
        ),
        text=[str(v) for v in chart_features.values()],
        textposition='outside',
        hovertemplate='%{y}: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="📊 Text Feature Analysis",
            font=dict(size=24, family="Inter"),
            
        ),
        xaxis_title="Count",
        height=400,
        margin=dict(l=150, r=50, t=60, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter"),
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=False)
    
    return fig


def create_confidence_gauge(confidence, prediction):
    color = '#10b981' if prediction == 1 else '#ef4444'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(family="Inter")
    )
    
    return fig


def main():
    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-title">🕵️ TruthLens</div>
            <div class="hero-subtitle">AI-Powered Fake News Detection</div>
            <div class="hero-description">
                Advanced machine learning technology to help you identify potentially misleading news content. 
                Analyze text patterns, linguistic features, and credibility indicators in real-time.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize NLTK and load model
    download_nltk_data()
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 How to Use")
        st.markdown("""
        1. **Enter the news headline** in the first field
        2. **Paste the article content** in the text area
        3. **Click 'Analyze Article'** to get results
        4. **Review the analysis** and confidence score
        5. **Check detailed features** below
        """)
        
        st.markdown("---")
        
        st.markdown("### 🎯 What We Analyze")
        st.markdown("""
        - **Language patterns** and writing style
        - **Punctuation usage** and formatting
        - **Emotional indicators** in text
        - **Structural characteristics** of content
        - **Statistical text features**
        """)
        
        st.markdown("---")
        
        st.markdown("### ⚠️ Important Notice")
        st.info("""
        This tool provides AI-assisted analysis only. Always:
        - Verify from multiple reliable sources
        - Check original publication details
        - Consider the publication's credibility
        - Use critical thinking
        """)
        
        # Add some usage statistics
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        
        st.markdown("---")
        st.markdown("### 📈 Session Stats")
        st.markdown(f"""
        - **Articles Analyzed**: {st.session_state.analysis_count}
        - **Session Started**: {datetime.now().strftime('%H:%M')}
        """)
    
    # Main Content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input Section
        st.markdown("""
            <div class="input-section">
                <div class="section-title">
                    📝 Enter News Content
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Input fields with better UX
        title_input = st.text_input(
            "📰 News Headline",
            placeholder="Enter the news headline here...",
            help="Paste the main headline or title of the news article"
        )
        
        content_input = st.text_area(
            "📄 Article Content",
            height=200,
            placeholder="Paste the full article content here...\n\nTip: The more content you provide, the more accurate the analysis will be.",
            help="Paste the complete text of the news article for best results"
        )
        
        # Analysis button with better styling
        analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
        with analyze_col2:
            analyze_button = st.button("🔍 Analyze Article", type="primary")
        
        # Analysis Results
        if analyze_button:
            if title_input.strip() and content_input.strip():
                st.session_state.analysis_count += 1
                combined_text = title_input + " " + content_input
                
                # Progress bar for better UX
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate processing steps
                status_text.text("🔄 Preprocessing text...")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                processed_input = enhanced_preprocessing(combined_text)
                
                status_text.text("🤖 Running AI analysis...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                prediction = model.predict([processed_input])[0]
                confidence = get_prediction_confidence(model, processed_input)
                
                status_text.text("📊 Generating results...")
                progress_bar.progress(75)
                time.sleep(0.5)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results with enhanced styling
                if prediction == 1:  # Real news
                    st.markdown(f'''
                    <div class="prediction-card real-news">
                        <span class="prediction-icon">✅</span>
                        <div class="prediction-text">LIKELY AUTHENTIC NEWS</div>
                        <div class="confidence-text">Confidence Level: {confidence:.1f}%</div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {confidence}%"></div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.success("✅ **Analysis Result**: This content appears to follow patterns typical of legitimate news articles.")
                    
                else:  # Fake news
                    st.markdown(f'''
                    <div class="prediction-card fake-news">
                        <span class="prediction-icon">⚠️</span>
                        <div class="prediction-text">POTENTIALLY MISLEADING</div>
                        <div class="confidence-text">Confidence Level: {confidence:.1f}%</div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {confidence}%"></div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    st.error("⚠️ **Analysis Result**: This content shows patterns often associated with misleading information. Please verify with reliable sources.")
                
                # Store results for sidebar display
                st.session_state.last_prediction = prediction
                st.session_state.last_confidence = confidence
                st.session_state.last_text = combined_text
                st.session_state.last_features = analyze_text_features(combined_text)
                
            else:
                st.error("❌ Please enter both a headline and article content to perform analysis.")
    
    with col2:
        st.markdown("### 📊 Analysis Dashboard")
        
        if 'last_text' in st.session_state and st.session_state.last_text.strip():
            # Confidence gauge
            confidence_fig = create_confidence_gauge(
                st.session_state.last_confidence, 
                st.session_state.last_prediction
            )
            st.plotly_chart(confidence_fig, use_container_width=True)
            
            # Feature analysis
            feature_fig = create_feature_chart(st.session_state.last_features)
            st.plotly_chart(feature_fig, use_container_width=True)
            
        else:
            st.info("📈 Analysis results will appear here after you analyze an article.")
    
    # Detailed Statistics Section
    if 'last_features' in st.session_state:
        st.markdown("### 📈 Detailed Text Statistics")
        
        # Create stats cards
        features = st.session_state.last_features
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{features['Word Count']}</div>
                <div class="stat-label">Total Words</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(' ')
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{features['Sentence Count']}</div>
                <div class="stat-label">Sentences</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{features['Exclamation Marks']}</div>
                <div class="stat-label">Exclamation Marks</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(' ')
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{features['All Caps Words']}</div>
                <div class="stat-label">All Caps Words</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{features['Average Word Length']}</div>
                <div class="stat-label">Avg Word Length</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(' ')
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{features['Average Sentence Length']}</div>
                <div class="stat-label">Avg Sentence Length</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Information Section
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 How Our AI Works</h4>
            <ul>
                <li>Advanced text preprocessing and feature extraction</li>
                <li>Machine learning classification with SVM algorithm</li>
                <li>Linguistic pattern and writing style analysis</li>
                <li>Confidence scoring based on model certainty</li>
                <li>Real-time processing and instant analysis</li>
                <li>Multi-feature detection for accurate results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🚨 Red Flag Indicators</h4>
            <ul>
                <li>Excessive punctuation marks (!!! ???)</li>
                <li>ALL CAPS words used for emphasis</li>
                <li>Emotional or highly sensational language</li>
                <li>Poor grammar and frequent spelling errors</li>
                <li>Lack of credible sources or citations</li>
                <li>Clickbait-style headlines and formatting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>✅ Verification Tips</h4>
            <ul>
                <li>Cross-check with multiple reliable news sources</li>
                <li>Verify author credentials and publication credibility</li>
                <li>Look for proper citations and source references</li>
                <li>Check publication date and content timeliness</li>
                <li>Use established fact-checking websites regularly</li>
                <li>Apply critical thinking to extraordinary claims</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #6b7280; font-size: 0.9rem; padding: 2rem;">
            <p>🕵️ <strong>TruthLens</strong> - AI-Powered Fake News Detection | Built with Streamlit & Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()