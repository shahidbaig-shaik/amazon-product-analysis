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
        """
        # Since the dataset is small (mock data), we can provide more comprehensive context
        # to ensure the LLM has the full picture.
        
        aspect_scores = pd.DataFrame(self.data['aspect_scores'])
        competitor_gaps = pd.DataFrame(self.data['competitor_gaps'])
        
        context_parts = []
        
        # 1. Provide all aspect scores (it's small enough)
        context_parts.append("--- Aspect Sentiment Scores (Scale: -1.0 to 1.0) ---")
        context_parts.append(aspect_scores.to_string(index=False))
        
        # 2. Provide relevant competitor gaps if comparison is implied
        if not competitor_gaps.empty:
            context_parts.append("\n--- Competitor Gaps (Positive = Target is better) ---")
            context_parts.append(competitor_gaps[['target_product', 'competitor', 'aspect', 'gap']].to_string(index=False))
            
        # 3. Add relevant review snippets based on keywords
        raw_data = pd.DataFrame(self.data['raw_processed_data'])
        keywords = query.lower().split()
        # Filter for reviews containing any of the keywords
        relevant_reviews = raw_data[raw_data['original_text'].str.contains('|'.join(keywords), case=False, na=False)]
        
        if not relevant_reviews.empty:
            # Limit to 5 most relevant reviews to avoid token limits
            context_parts.append("\n--- Relevant Customer Review Snippets ---")
            context_parts.append(relevant_reviews[['product_name', 'original_text']].head(5).to_string(index=False))
            
        return "\n".join(context_parts)
