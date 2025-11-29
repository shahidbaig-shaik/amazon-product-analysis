import streamlit as st
import google.generativeai as genai
import os
import pandas as pd

class ChatInterface:
    def __init__(self, data):
        self.data = data
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def render(self):
        # API Key Input if not found
        if not self.api_key:
            self.api_key = st.text_input("Enter Google API Key", type="password")
            if not self.api_key:
                st.warning("Please enter your Google API Key to use the chatbot.")
                return
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask about the product analysis..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                response_text = self.generate_response(prompt)
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

    def generate_response(self, query):
        """
        Simple RAG implementation:
        1. Retrieve relevant context from data
        2. Prompt Gemini with context and query
        """
        context = self.retrieve_context(query)
        
        # Dynamically find a supported model
        supported_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    supported_models.append(m.name)
        except Exception as e:
            print(f"Error listing models: {e}")

        # Prefer gemini-1.5-flash, then gemini-pro, then whatever is available
        model_name = 'models/gemini-1.5-flash' # Default fallback
        
        if supported_models:
            # print(f"Available models: {supported_models}") # Debugging
            if 'models/gemini-1.5-flash' in supported_models:
                model_name = 'models/gemini-1.5-flash'
            elif 'models/gemini-pro' in supported_models:
                model_name = 'models/gemini-pro'
            else:
                model_name = supported_models[0] # Pick the first available one
        
        # print(f"Using model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        You are an intelligent business analyst assistant for an Amazon Product Analysis dashboard.
        Your goal is to answer the user's specific question based *only* on the provided data context.
        
        **Instructions:**
        1.  **Be Direct**: Answer the specific question asked. Do not summarize everything unless asked.
        2.  **Use Data**: Cite specific sentiment scores (range -1 to 1) to back up your claims.
        3.  **Compare**: If the user asks for a comparison, explicitly mention the gap.
        4.  **Tone**: Professional, concise, and insightful.
        
        **Data Context:**
        {context}
        
        **User Question:** {query}
        
        **Answer:**
        """
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"

    def retrieve_context(self, query):
        """
        Retrieves relevant insights based on the query.
        Includes SMART MAPPING to guess Product IDs from user keywords (e.g. 'Oatmeal' -> 'B00...')
        """
        aspect_scores = pd.DataFrame(self.data['aspect_scores'])
        competitor_gaps = pd.DataFrame(self.data['competitor_gaps'])
        raw_data = pd.DataFrame(self.data['raw_processed_data'])
        
        context_parts = []
        
        # --- SMART MAPPING LOGIC ---
        # Search for query keywords in review text to map "Oatmeal" -> "B00..."
        query_terms = query.lower().split()
        mapping_hints = []
        
        if not raw_data.empty:
            for term in query_terms:
                if len(term) < 4: continue # Skip small words
                # Find products where this term appears in reviews
                matches = raw_data[raw_data['original_text'].str.contains(term, case=False, na=False)]
                if not matches.empty:
                    # Get the most frequent product for this term
                    top_product = matches['product_name'].mode()
                    if not top_product.empty:
                        prod_id = top_product[0]
                        mapping_hints.append(f"NOTE: User term '{term}' likely refers to Product ID: {prod_id}")
        
        if mapping_hints:
            context_parts.append("--- ID MAPPING HINTS (IMPORTANT: Use these to match user names to Product IDs) ---")
            context_parts.append("\n".join(mapping_hints))
        # ---------------------------
        
        # Add aspect scores summary
        context_parts.append("\n--- Aspect Scores (by Product ID) ---")
        context_parts.append(aspect_scores.to_string(index=False))
        
        # Add competitor gaps summary
        if not competitor_gaps.empty:
            context_parts.append("\n--- Competitor Gaps ---")
            context_parts.append(competitor_gaps.to_string(index=False))
            
        # Add some raw reviews if specific keywords are mentioned
        keywords = query.lower().split()
        relevant_reviews = raw_data[raw_data['original_text'].str.contains('|'.join(keywords), case=False, na=False)].head(5)
        
        if not relevant_reviews.empty:
            context_parts.append("\n--- Relevant Review Snippets ---")
            context_parts.append(relevant_reviews[['product_name', 'original_text']].to_string(index=False))
            
        return "\n".join(context_parts)
