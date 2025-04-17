import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import plotly.graph_objects as go
import numpy as np

def show_llm():
    st.title("📚 LLM Document Analysis")
    
    # Model initialization
    @st.cache_resource
    def load_model():
        model_name = "facebook/bart-large-cnn"  # Using BART model which is available in PyTorch
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float32
        )
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            min_length=50,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # File upload
    st.subheader("Document Upload")
    file_type = st.radio("Select file type:", ["PDF", "CSV"])
    uploaded_file = st.file_uploader(
        "Upload your document", 
        type=['pdf'] if file_type == "PDF" else ['csv']
    )
    
    if uploaded_file:
        try:
            with st.spinner("Processing document..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type.lower()}') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Load and process document
                if file_type == "PDF":
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = CSVLoader(tmp_path)
                
                documents = loader.load()
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.split_documents(documents)
                
                # Create vector store
                vectorstore = FAISS.from_documents(chunks, embeddings)
                
                # Create retrieval chain
                with st.spinner("Loading model... This may take a moment."):
                    llm = load_model()
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        ),
                        return_source_documents=True,
                        verbose=True
                    )
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            st.subheader("Chat Interface")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "metadata" in message:
                        with st.expander("Response Metadata"):
                            st.json(message["metadata"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your document"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Get chat history
                        chat_history = [(m["content"], r["content"]) 
                                      for m, r in zip(st.session_state.messages[::2], 
                                                    st.session_state.messages[1::2])]
                        
                        try:
                            # Get response from chain
                            result = qa_chain(
                                {"question": prompt, "chat_history": chat_history}
                            )
                            
                            response = result["answer"]
                            st.markdown(response)
                            
                            # Get similarity scores
                            similarity_results = vectorstore.similarity_search_with_score(prompt)
                            relevance_scores = [float(score) for _, score in similarity_results]
                            
                            # Add metadata
                            metadata = {
                                "source_documents": [
                                    {
                                        "page_content": doc.page_content[:200] + "...",
                                        "source": doc.metadata.get("source", "Unknown"),
                                        "page": doc.metadata.get("page", 0)
                                    }
                                    for doc in result["source_documents"]
                                ],
                                "relevance_scores": relevance_scores
                            }
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "metadata": metadata
                            })
                            
                            # Show confidence
                            confidence = float(np.mean(relevance_scores))
                            st.metric("Response Confidence", f"{confidence:.2%}")
                            
                        except Exception as e:
                            st.error(f"Error processing your question: {str(e)}")
                            st.info("Please try rephrasing your question or uploading a different document.")
            
            # Cleanup
            os.unlink(tmp_path)
            
            # Add visualization of conversation
            if st.session_state.messages:
                st.subheader("Conversation Analysis")
                
                # Create visualization using plotly
                fig = go.Figure()
                
                # Add conversation flow
                user_msgs = [msg for msg in st.session_state.messages if msg["role"] == "user"]
                assistant_msgs = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
                
                if user_msgs:  # Only add trace if there are messages
                    fig.add_trace(go.Scatter(
                        x=list(range(len(user_msgs))),
                        y=[len(msg["content"].split()) for msg in user_msgs],
                        name="User Messages",
                        mode="lines+markers"
                    ))
                
                if assistant_msgs:  # Only add trace if there are messages
                    fig.add_trace(go.Scatter(
                        x=list(range(len(assistant_msgs))),
                        y=[len(msg["content"].split()) for msg in assistant_msgs],
                        name="Assistant Responses",
                        mode="lines+markers"
                    ))
                    
                    # Add confidence scores if available
                    confidences = [
                        float(np.mean(msg.get("metadata", {}).get("relevance_scores", [0])))
                        for msg in assistant_msgs
                    ]
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(assistant_msgs))),
                        y=confidences,
                        name="Response Confidence",
                        mode="lines+markers",
                        yaxis="y2"
                    ))
                
                # Update layout
                fig.update_layout(
                    title="Conversation Length and Confidence Over Time",
                    xaxis_title="Message Number",
                    yaxis_title="Word Count",
                    yaxis2=dict(
                        title="Confidence Score",
                        overlaying="y",
                        side="right",
                        range=[0, 1]
                    ),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        st.info("Please upload a document to begin the analysis.")
        
    # Add architecture explanation
    with st.expander("How it works"):
        st.markdown("""
        This LLM-powered document analysis tool uses the following components:
        
        1. **Document Processing**:
           - Loads PDF/CSV documents
           - Splits them into manageable chunks
           - Maintains document structure and metadata
        
        2. **Embedding & Retrieval**:
           - Uses sentence-transformers for document embedding
           - FAISS vector store for efficient similarity search
           - Retrieves relevant context for each question
        
        3. **Language Model**:
           - Uses BART for natural language understanding
           - Generates coherent and contextual responses
           - Maintains conversation history
        
        4. **Analysis & Visualization**:
           - Tracks conversation metrics
           - Displays confidence scores
           - Visualizes interaction patterns
        """)

if __name__ == "__main__":
    show_llm() 