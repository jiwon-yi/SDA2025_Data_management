"""
SDA 2025 - Data Management, Data Visualisation & Text Mining
Projet Final: Analyse Interactive des Reviews Snapchat
Auteur: Jiwon Yi
Date: Juin 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# WordCloud import
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="Snapchat Reviews Analytics - SDA 2025",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour style Snapchat professionnel
st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
    }
    
    .snapchat-header {
        background: linear-gradient(135deg, #FFFC00 0%, #FFD700 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .snapchat-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #000000;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.3);
    }
    
    .snapchat-subtitle {
        font-size: 1.2rem;
        color: #333333;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    .metric-container {
        background: #F8F9FA;
        border: 2px solid #E9ECEF;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2C3E50;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6C757D;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF5252 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 600;
        border-left: 5px solid #C62828;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #FFB74D 0%, #FFA726 100%);
        color: #333;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 600;
        border-left: 5px solid #F57C00;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #66BB6A 0%, #4CAF50 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-weight: 600;
        border-left: 5px solid #2E7D32;
    }
    
    .section-header {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-size: 1.4rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .snapchat-ghost {
        font-size: 3rem;
        color: #FFFC00;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-right: 1rem;
    }
    
    .executive-summary {
        background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .summary-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: #FFFC00;
    }
    
    .summary-content {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        border: 2px solid #FFFC00;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .insight-title {
        color: #2C3E50;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .insight-content {
        color: #495057;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .data-quality-box {
        background: #E8F5E8;
        border: 2px solid #4CAF50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .methodology-box {
        background: #E3F2FD;
        border: 2px solid #2196F3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="snapchat-header">
    <div style="display: flex; align-items: center; justify-content: center;">
        <span class="snapchat-ghost">👻</span>
        <div>
            <h1 class="snapchat-title">SNAPCHAT REVIEWS ANALYTICS</h1>
            <p class="snapchat-subtitle">Projet SDA 2025 - Data Management & Visualisation</p>
            <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 1rem;">Jiwon Yi | Analyse Interactive de 200K+ Reviews</p>
        </div>
        <span class="snapchat-ghost">👻</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Créer des données d'exemple
@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n_reviews = 1000
    
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', periods=n_reviews)
    data = {
        'reviewId': [f'review_{i}' for i in range(n_reviews)],
        'userName': [f'user_{i}' for i in range(n_reviews)],
        'score': np.random.choice([1, 2, 3, 4, 5], n_reviews, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
        'thumbsUpCount': np.random.poisson(3, n_reviews),
        'at': dates,
        'appVersion': np.random.choice(['12.1.0', '12.2.0', '12.3.0'], n_reviews),
        'content': ['Great app! Love the filters.' if i % 3 == 0 
                   else 'App crashes frequently.' if i % 3 == 1 
                   else 'Average experience, could be better.' for i in range(n_reviews)]
    }
    
    df = pd.DataFrame(data)
    df['content_length'] = df['content'].str.len()
    df['sentiment_score'] = np.where(df['score'] >= 4, 0.8, 
                                   np.where(df['score'] <= 2, -0.5, 0.0)) + np.random.normal(0, 0.2, n_reviews)
    df['sentiment_category'] = np.where(df['sentiment_score'] > 0.3, 'Positif',
                                      np.where(df['sentiment_score'] < -0.3, 'Négatif', 'Neutre'))
    df['length_category'] = pd.cut(df['content_length'], bins=[0, 20, 50, 200], 
                                  labels=['Court', 'Moyen', 'Long'])
    df['period_category'] = pd.cut(df['at'], bins=3, labels=['Ancien', 'Récent', 'Très récent'])
    df['engagement_ratio'] = df['thumbsUpCount'] / (df['content_length'] + 1)
    df['engagement_category'] = pd.cut(df['engagement_ratio'], bins=3, 
                                     labels=['Faible', 'Moyen', 'Fort'])
    
    return df

# Charger les données
try:
    df = pd.read_csv("snapchat_reviews_clean.csv")
    df['at'] = pd.to_datetime(df['at'])
    
    # Ajouter colonnes manquantes si nécessaire
    if 'sentiment_category' not in df.columns:
        df['sentiment_category'] = df['score'].map({1: 'Négatif', 2: 'Négatif', 3: 'Neutre', 4: 'Positif', 5: 'Positif'})
    if 'content_length' not in df.columns:
        df['content_length'] = df['content'].str.len()
    if 'length_category' not in df.columns:
        df['length_category'] = pd.cut(df['content_length'], bins=[0, 50, 120, 300], labels=['Court', 'Moyen', 'Long'])
    if 'period_category' not in df.columns:
        df['period_category'] = pd.cut(df['at'], bins=3, labels=['Ancien', 'Récent', 'Très récent'])
    
    st.success(f"✅ Dataset chargé: {df.shape[0]:,} reviews")
except FileNotFoundError:
    df = create_sample_data()
    st.warning("⚠️ Utilisation de données d'exemple")

# Sidebar filtres
st.sidebar.header("🎛️ Filtres")

period_filter = st.sidebar.multiselect(
    "📅 Périodes:",
    options=df['period_category'].unique(),
    default=df['period_category'].unique()
)

score_filter = st.sidebar.slider("⭐ Scores:", 1, 5, (1, 5))

sentiment_filter = st.sidebar.multiselect(
    "😊 Sentiment:",
    options=df['sentiment_category'].unique(),
    default=df['sentiment_category'].unique()
)

length_filter = st.sidebar.multiselect(
    "📝 Longueur:",
    options=df['length_category'].unique(),
    default=df['length_category'].unique()
)

# Appliquer filtres
df_filtered = df[
    (df['period_category'].isin(period_filter)) &
    (df['score'] >= score_filter[0]) &
    (df['score'] <= score_filter[1]) &
    (df['sentiment_category'].isin(sentiment_filter)) &
    (df['length_category'].isin(length_filter))
]

st.sidebar.write(f"📊 {len(df_filtered):,} reviews sélectionnées")

# Section 1: Dataset
st.header("📊 Présentation du Dataset")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Informations")
    st.write(f"**Source:** Google Play Store")
    st.write(f"**App:** Snapchat")
    st.write(f"**Reviews:** {len(df):,}")
    st.write(f"**Variables:** {len(df.columns)}")

with col2:
    st.subheader("🔍 Variables")
    st.write("**Originales:** reviewId, userName, content, score, thumbsUpCount, at, appVersion")
    st.write("**Créées:** sentiment_category, content_length, length_category, period_category")

# Métriques avec alertes business
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📱 Total", f"{len(df):,}")
with col2:
    avg_score = df_filtered['score'].mean()
    st.metric("⭐ Score Moyen", f"{avg_score:.2f}")
with col3:
    st.metric("📝 Longueur Moy.", f"{df_filtered['content_length'].mean():.0f}")
with col4:
    st.metric("👍 Thumbs Moy.", f"{df_filtered['thumbsUpCount'].mean():.1f}")

# 🚀 NOUVEAUTÉ: Alertes Business Simples
st.subheader("🚨 Alertes Business")

negative_rate = (df_filtered['sentiment_category'] == 'Négatif').sum() / len(df_filtered) * 100
critical_reviews = (df_filtered['score'] <= 2).sum() / len(df_filtered) * 100

col1, col2, col3 = st.columns(3)
with col1:
    if avg_score < 3.5:
        st.error(f"🚨 Score critique: {avg_score:.2f}")
    elif avg_score < 4.0:
        st.warning(f"⚠️ Score à surveiller: {avg_score:.2f}")
    else:
        st.success(f"✅ Score satisfaisant: {avg_score:.2f}")

with col2:
    if negative_rate > 25:
        st.error(f"🚨 Taux négatif élevé: {negative_rate:.1f}%")
    elif negative_rate > 15:
        st.warning(f"⚠️ Taux négatif: {negative_rate:.1f}%")
    else:
        st.success(f"✅ Sentiment stable: {negative_rate:.1f}%")

with col3:
    if critical_reviews > 20:
        st.error(f"🚨 Reviews critiques: {critical_reviews:.1f}%")
    elif critical_reviews > 10:
        st.warning(f"⚠️ Reviews critiques: {critical_reviews:.1f}%")
    else:
        st.success(f"✅ Reviews critiques: {critical_reviews:.1f}%")

# Statistiques
st.header("📈 Statistiques Descriptives")
numeric_cols = ['score', 'thumbsUpCount', 'content_length']
if 'sentiment_score' in df_filtered.columns:
    numeric_cols.append('sentiment_score')
st.dataframe(df_filtered[numeric_cols].describe().round(2))

# Visualisations
st.header("📊 Visualisations")

# 1. Distribution des scores
st.subheader("1️⃣ Distribution des Scores")
fig1 = px.histogram(df_filtered, x='score', nbins=5, title="Distribution des Scores")
st.plotly_chart(fig1, use_container_width=True)

# 2. Évolution temporelle - VERSION SIMPLE ET SÛRE
st.subheader("2️⃣ Évolution Temporelle")
if len(df_filtered) > 0:
    # Créer une version simple et sûre
    df_temp = df_filtered.copy()
    df_temp['month'] = df_temp['at'].dt.to_period('M').astype(str)
    monthly = df_temp.groupby('month')['score'].mean().reset_index()
    
    # Utiliser plotly express qui est plus stable
    fig2 = px.line(monthly, x='month', y='score', 
                   title="Évolution des Scores Moyens",
                   labels={'month': 'Mois', 'score': 'Score Moyen'})
    fig2.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

# 3. Scatter plot
st.subheader("3️⃣ Score vs Engagement")
fig3 = px.scatter(df_filtered.sample(min(1000, len(df_filtered))), 
                  x='score', y='thumbsUpCount', 
                  color='sentiment_category',
                  title="Score vs ThumbsUp")
st.plotly_chart(fig3, use_container_width=True)

# 4. Box plot
st.subheader("4️⃣ Longueur par Sentiment")
fig4 = px.box(df_filtered, x='sentiment_category', y='content_length',
              title="Distribution Longueur par Sentiment")
st.plotly_chart(fig4, use_container_width=True)

# 5. Corrélation
st.subheader("5️⃣ Corrélations")
corr_data = df_filtered[numeric_cols].corr()
fig5 = px.imshow(corr_data, text_auto=True, title="Matrice de Corrélation")
st.plotly_chart(fig5, use_container_width=True)

# 🚀 NOUVEAUTÉ: Analyse par Version d'App (si disponible)
if 'appVersion' in df_filtered.columns:
    st.subheader("6️⃣ Performance par Version")
    
    version_stats = df_filtered.groupby('appVersion').agg({
        'score': 'mean',
        'reviewId': 'count',
        'thumbsUpCount': 'mean'
    }).round(2)
    version_stats.columns = ['Score_Moyen', 'Nb_Reviews', 'Engagement_Moyen']
    
    col1, col2 = st.columns(2)
    with col1:
        fig_version = px.bar(version_stats.reset_index(), 
                           x='appVersion', y='Score_Moyen',
                           title="Score Moyen par Version")
        st.plotly_chart(fig_version, use_container_width=True)
    
    with col2:
        st.write("**📊 Tableau Performance par Version**")
        st.dataframe(version_stats)
        
        # Insight automatique
        best_version = version_stats['Score_Moyen'].idxmax()
        worst_version = version_stats['Score_Moyen'].idxmin()
        st.info(f"💡 **Insight:** Meilleure version: {best_version} ({version_stats.loc[best_version, 'Score_Moyen']:.2f})")

# Text Mining
st.header("🔤 Text Mining")

article_text = """
Les plateformes mobiles révolutionnent communication moderne créativité innovation.
Fonctionnalités disparition messages sécurité confidentialité unique.
Filtres réalité augmentée divertissement engagement communauté active.
Technologie intelligence artificielle reconnaissance faciale effets visuels.
Cependant problèmes persistent addiction cyberharcèlement modération contenu.
Développeurs améliorent constamment expérience sécurité protection données.
Avenir dépend innovation technologique respect vie privée utilisateurs.
Concurrence pousse amélioration continue fonctionnalités interface.
"""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Zàâäéèêëïîôöùûüÿñç\s]', '', text)
    
    # Liste étendue de mots vides français
    stop_words = [
        'le', 'de', 'un', 'à', 'être', 'et', 'en', 'avoir', 'que', 'pour', 
        'dans', 'ce', 'il', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout',
        'plus', 'par', 'grand', 'son', 'du', 'comme', 'des', 'les', 'la',
        'cette', 'ces', 'leur', 'leurs', 'nous', 'vous', 'ils', 'elles',
        'ont', 'sont', 'était', 'été', 'fait', 'peut', 'très', 'bien',
        'encore', 'même', 'aussi', 'autre', 'autres', 'tous', 'toutes',
        'certains', 'certaines', 'plusieurs', 'nombreux', 'nombreuses',
        'utilisateurs', 'utilisateur', 'réseaux', 'réseau', 'sociaux',
        'social', 'applications', 'application', 'plateforme', 'plateformes', 
        'snapchat', 'snap', 'cependant'  # Mots génériques exclus
    ]
    
    words = [word for word in text.split() if word not in stop_words and len(word) > 3]
    return ' '.join(words)

processed = preprocess_text(article_text)
st.text_area("Texte prétraité:", processed, height=100)

# WordCloud ou alternative
if WORDCLOUD_AVAILABLE:
    try:
        wc = WordCloud(width=800, height=400, background_color='white',
                      colormap='viridis', max_words=40).generate(processed)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.title('WordCloud - Analyse Technologique')
        st.pyplot(fig)
        plt.close()
    except:
        st.error("Erreur WordCloud")
else:
    st.info("Installez wordcloud: pip install wordcloud")

# Fréquence des mots
words = processed.split()
word_freq = Counter(words).most_common(10)
if word_freq:
    word_df = pd.DataFrame(word_freq, columns=['Mot', 'Fréquence'])
    fig_words = px.bar(word_df, x='Mot', y='Fréquence', title="Mots Fréquents")
    st.plotly_chart(fig_words, use_container_width=True)

# 🚀 NOUVEAUTÉ: Analyse Business des Reviews
st.header("💼 Insights Business")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Points Forts")
    
    strengths = []
    if avg_score >= 4.0:
        strengths.append(f"✅ Score utilisateur excellent ({avg_score:.2f}/5)")
    if negative_rate < 20:
        strengths.append(f"✅ Sentiment négatif maîtrisé ({negative_rate:.1f}%)")
    if df_filtered['thumbsUpCount'].mean() > 2:
        strengths.append(f"✅ Engagement satisfaisant")
    
    if not strengths:
        strengths.append("🎯 Opportunités d'amélioration identifiées")
    
    for strength in strengths:
        st.write(strength)

with col2:
    st.subheader("🎯 Recommandations")
    
    recommendations = []
    if avg_score < 4.0:
        recommendations.append("🔧 Améliorer l'expérience utilisateur globale")
    if negative_rate > 20:
        recommendations.append("📞 Traiter proactivement les feedbacks négatifs")
    if critical_reviews > 15:
        recommendations.append("🚨 Résoudre les problèmes critiques urgents")
    
    if not recommendations:
        recommendations.append("✅ Maintenir l'excellence actuelle")
    
    for rec in recommendations:
        st.write(rec)

# 🚀 NOUVEAUTÉ: Export Business Report
st.subheader("📊 Rapport Exécutif")

if st.button("📈 Générer Résumé Business"):
    business_summary = f"""
    ## 📱 RAPPORT SNAPCHAT - RÉSUMÉ EXÉCUTIF
    
    **📊 KPI Principaux:**
    - Score moyen: {avg_score:.2f}/5 {'✅' if avg_score >= 4.0 else '⚠️'}
    - Sentiment négatif: {negative_rate:.1f}% {'✅' if negative_rate < 20 else '⚠️'}
    - Reviews analysées: {len(df_filtered):,}
    
    **🎯 Status Global:** {'Excellent' if avg_score >= 4.0 and negative_rate < 15 else 'Bon' if avg_score >= 3.5 else 'À améliorer'}
    
    **💡 Action Prioritaire:** {'Maintenir la qualité' if avg_score >= 4.0 and negative_rate < 20 else 'Plan d\'amélioration immédiat'}
    """
    
    st.markdown(business_summary)

# Conclusions
st.header("🎯 Conclusions")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Insights Principaux")
    avg_final_score = df_filtered['score'].mean()
    positive_final_rate = (df_filtered['sentiment_category']=='Positif').sum()/len(df_filtered)*100
    
    st.write(f"• Score moyen: **{avg_final_score:.2f}/5**")
    st.write(f"• Reviews positives: **{positive_final_rate:.1f}%**")
    
    # Distribution réelle des scores
    score_dist = df_filtered['score'].value_counts().sort_index()
    most_common_score = score_dist.idxmax()
    st.write(f"• Score le plus fréquent: **{most_common_score}⭐** ({score_dist[most_common_score]} reviews)")

with col2:
    st.subheader("🔍 Observations Réelles")
    
    if avg_final_score >= 4.0:
        st.write("• Performance globalement satisfaisante")
    elif avg_final_score >= 3.5:
        st.write("• Performance moyenne, améliorations possibles")
    else:
        st.write("• Performance en dessous des attentes")
    
    if positive_final_rate >= 50:
        st.write("• Sentiment majoritairement positif")
    else:
        st.write(f"• Sentiment mitigé ({positive_final_rate:.1f}% positif)")
    
    # Corrélation réelle
    corr_score_thumbs = df_filtered['score'].corr(df_filtered['thumbsUpCount'])
    st.write(f"• Corrélation score-engagement: **{corr_score_thumbs:.2f}**")

st.markdown("---")
