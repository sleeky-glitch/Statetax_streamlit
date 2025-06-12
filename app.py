import streamlit as st
import openai
import pinecone
from pinecone import Pinecone
import os
from typing import List, Dict
import time

# Page configuration
st.set_page_config(
    page_title="State Tax Department Assistant",
    page_icon="🏛️",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pinecone_client" not in st.session_state:
    st.session_state.pinecone_client = None
if "index" not in st.session_state:
    st.session_state.index = None

def initialize_clients():
    """Initialize Pinecone and OpenAI clients"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index("gstindex")

        # Initialize OpenAI
        openai.api_key = st.secrets["OPENAI_API_KEY"]

        st.session_state.pinecone_client = pc
        st.session_state.index = index

        return True
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return False

def get_embedding(text: str) -> List[float]:
    """Get embedding for text using OpenAI text-embedding-3-small"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return []

def search_similar_documents(query: str, top_k: int = 5) -> List[Dict]:
    """Search for similar documents in Pinecone index"""
    try:
        # Get query embedding
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []

        # Search in Pinecone
        results = st.session_state.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        return results.matches
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

def generate_response(query: str, context_docs: List[Dict]) -> str:
    """Generate response using OpenAI with retrieved context"""
    try:
        # Prepare context from retrieved documents
        context = ""
        for i, doc in enumerate(context_docs):
            metadata = doc.get('metadata', {})
            # Extract text content from metadata (adjust key names based on your data structure)
            text_content = metadata.get('text', metadata.get('content', str(metadata)))
            context += f"Document {i+1}: {text_content}\n\n"

        # Enhanced system prompt for state tax department
        system_prompt = """You are an expert AI assistant for the State Tax Department, specializing in providing accurate, comprehensive, and authoritative information about state tax laws, regulations, procedures, and compliance requirements.

Your role and responsibilities:
- Provide precise answers based on official tax documents, regulations, and guidelines
- Explain complex tax concepts in clear, understandable language
- Guide taxpayers through procedures, forms, and compliance requirements
- Reference specific tax codes, sections, and regulations when applicable
- Maintain a professional, helpful, and authoritative tone
- Ensure all information is current and legally accurate

Guidelines for responses:
1. Always base your answers on the provided context documents
2. If information is not available in the context, clearly state this limitation
3. For complex tax matters, recommend consulting with a tax professional or contacting the department directly
4. Include relevant form numbers, deadlines, and procedural steps when applicable
5. Explain both the requirements and the consequences of non-compliance
6. Use official terminology and cite specific regulations when possible
7. Provide step-by-step guidance for procedures
8. Highlight important deadlines, penalties, and compliance requirements

Important disclaimers to include when appropriate:
- Tax laws may change; verify current requirements
- Individual circumstances may affect tax obligations
- Professional tax advice may be necessary for complex situations
- Contact the State Tax Department for case-specific guidance"""

        user_prompt = f"""Based on the official state tax documents and regulations provided below, please answer the taxpayer's question with accuracy and authority.

OFFICIAL TAX DOCUMENTS AND REGULATIONS:
{context}

TAXPAYER QUESTION: {query}

Please provide a comprehensive, authoritative response that:
1. Directly answers the question using the official documentation
2. Cites specific regulations, forms, or procedures mentioned in the documents
3. Explains any relevant deadlines, requirements, or compliance obligations
4. Provides step-by-step guidance if applicable
5. Mentions any forms that need to be filed or procedures to follow
6. Highlights important warnings about penalties or compliance issues
7. Suggests next steps or additional resources if needed

If the provided documents don't contain sufficient information to fully answer the question, clearly state what information is missing and recommend contacting the State Tax Department directly."""

        # Generate response using OpenAI
        response = openai.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.3  # Lower temperature for more consistent, factual responses
        )

        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating a response. Please try again or contact the State Tax Department directly for assistance."

def main():
    st.title("🏛️ State Tax Department Assistant")
    st.markdown("""
    **Welcome to the Official State Tax Department AI Assistant**
    
    Get accurate information about state tax laws, regulations, filing procedures, and compliance requirements. 
    This assistant is powered by official tax documents and regulations.
    
    ⚠️ **Important Notice**: This assistant provides general guidance based on official documents. For complex tax matters or case-specific advice, please consult with a tax professional or contact the State Tax Department directly.
    """)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Initialize clients button
        if st.button("Initialize Connections"):
            with st.spinner("Connecting to tax database..."):
                if initialize_clients():
                    st.success("✅ Connected to tax database successfully!")
                else:
                    st.error("❌ Failed to connect to tax database")

        # Connection status
        if st.session_state.index is not None:
            st.success("🟢 Connected to Tax Database")
        else:
            st.warning("🟡 Not connected to tax database")

        # Search parameters
        st.subheader("Search Parameters")
        top_k = st.slider("Number of documents to retrieve", 1, 10, 5)

        # Helpful links
        st.subheader("📋 Quick Links")
        st.markdown("""
        - [Tax Forms & Publications](#)
        - [Payment Portal](#)
        - [File a Return](#)
        - [Contact Information](#)
        - [Tax Calendar](#)
        """)

        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Official Sources Referenced"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**Official Document {i+1}** (Relevance Score: {source.get('score', 'N/A'):.3f})")
                        metadata = source.get('metadata', {})
                        st.write(f"Content: {str(metadata)[:300]}...")

    # Chat input with examples
    st.markdown("### 💬 Ask Your Tax Question")
    st.markdown("**Example questions:**")
    st.markdown("""
    - "What are the filing deadlines for quarterly sales tax returns?"
    - "How do I register my business for state tax purposes?"
    - "What documentation is required for a tax exemption certificate?"
    - "What are the penalties for late tax filing?"
    """)

    if prompt := st.chat_input("Ask about state tax laws, procedures, forms, deadlines, or compliance requirements..."):
        # Check if clients are initialized
        if st.session_state.index is None:
            st.error("⚠️ Please initialize connections first using the sidebar to access the tax database.")
            return

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching official tax documents and generating response..."):
                # Search for similar documents
                similar_docs = search_similar_documents(prompt, top_k)

                if similar_docs:
                    # Generate response
                    response = generate_response(prompt, similar_docs)

                    # Display response
                    st.markdown(response)

                    # Add assistant message to chat history with sources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": similar_docs
                    })

                    # Show sources
                    with st.expander("📚 Official Sources Referenced"):
                        for i, doc in enumerate(similar_docs):
                            st.write(f"**Official Document {i+1}** (Relevance Score: {doc.get('score', 'N/A'):.3f})")
                            metadata = doc.get('metadata', {})
                            st.write(f"Content: {str(metadata)[:300]}...")
                else:
                    error_msg = """I couldn't find relevant information in our official tax database to answer your question. 
                    
Please consider:
- Rephrasing your question with more specific tax terms
- Contacting the State Tax Department directly at [phone number]
- Visiting our website at [website URL]
- Scheduling an appointment with a tax specialist"""
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Footer with disclaimer
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This AI assistant provides general information based on official state tax documents. Tax laws and regulations may change. 
    For the most current information and case-specific advice, please contact the State Tax Department directly or consult with a qualified tax professional.
    """)

if __name__ == "__main__":
    main()
