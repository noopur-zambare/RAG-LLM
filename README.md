## Demo
RAG (Retrieval-Augmented Generation) combines retrieval and generation to improve the performance of language models, especially for tasks like question answering, summarization, and more. The core idea is to use external documents which acts like knowledge source to retrieve relevant information before generating a response. This approach allows models to handle larger knowledge bases and provide more accurate and relevant answers.

## To install Ollama and run the application
Download it from the official website, then run the following command.

```
ollama run deepseek-r1:1.5b
ollama run 
ollama run MistralSmall3
```

```
streamlit run app.py
```

## Key Libraries Used
- Streamlit for creating the web interface.
- LangChain for document loading, text splitting, and creating a vector database.
- Ollama for using pre-trained language models to generate document embeddings and provide responses.

## Instructions:
- Upload a PDF document.
- Ask questions about the document once it's processed.
- Toggle between themes for a better visual experience.
