import streamlit as st
import json
import os
from visuals import plot_aspect_radar, plot_competitor_gap
from chat import ChatInterface

st.set_page_config(page_title="Amazon Product Analysis", layout="wide")

def load_data():
    # Look for insights.json in the same directory
    path = "insights.json"
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def main():
    st.title("AI-Driven Product Analysis System")
    
    data = load_data()
    if not data:
        st.error("No insights found. Please ensure insights.json is uploaded.")
        return

    # Sidebar for Product Selection
    products = sorted(list(set([item['product_name'] for item in data['aspect_scores']])))
    selected_product = st.sidebar.selectbox("Select Product", products)
    
    # Main Dashboard
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Analysis for {selected_product}")
        
        # 1. Aspect Scores Radar Chart
        st.markdown("### Aspect Sentiment Scores")
        fig_radar = plot_aspect_radar(data['aspect_scores'], selected_product)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # 2. Competitor Gaps
        st.markdown("### Competitor Gap Analysis")
        fig_gap = plot_competitor_gap(data['competitor_gaps'], selected_product)
        if fig_gap:
            st.plotly_chart(fig_gap, use_container_width=True)
        else:
            st.info("No competitor data available for comparison.")

    with col2:
        st.subheader("AI Assistant")
        chat = ChatInterface(data)
        chat.render()

if __name__ == "__main__":
    main()
