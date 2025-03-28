# Conversational AI for Documents Using RAG and LLM
RAG (Retrieval-Augmented Generation) combines retrieval and generation to improve the performance of language models, especially for tasks like question answering, summarization, and more. The core idea is to use external documents, which act like knowledge sources, to retrieve relevant information before generating a response. This approach allows models to handle larger knowledge bases and provide more accurate and relevant answers.

## Demo
https://github.com/user-attachments/assets/55016a4b-83bb-4162-9367-804bbbaef0a7

## To install Ollama and run the application
Download it from the official website, then run the following command.

```
ollama run deepseek-r1:1.5b
ollama run llama3.3
ollama run mixtral
```

```
streamlit run app.py
```

## Key Libraries Used
- Streamlit for creating the web interface.
- LangChain for document loading, text splitting, and creating a vector database.
- Ollama for using pre-trained language models to generate document embeddings and provide responses.

## Instructions
- Upload a PDF document.
- Ask questions about the document once it's processed.
- Toggle between themes for a better visual experience.
