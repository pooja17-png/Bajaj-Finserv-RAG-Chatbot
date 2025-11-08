# Bajaj-Finserv-RAG-Chatbot
This project is a great example of how a chatbot can be used to extract actionable insights from complex financial reports, specifically fund factsheets, and present them in a digestible format for users.

Introducing RAGs:

Retrieval Augmented Generation often termed as RAG is an AI technique that combines a large language model (LLM) with an external knowledge base to create more accurate and context-specific responses. It works by first retrieving relevant information from external sources, like a database or document, and then using that retrieved information it combines with the LLM's response, improve its relevance and accuracy. This allows the AI to use up-to-date and specific data it wasn't originally trained on, without needing costly contant training or fine-tuning.

RAG is basically a combination of two techniques-- Information Retrieval + Text Generation

RAG is mainly divided into 4 steps: 1)Indexing 2)Retreival 3)Augmentation 4)Generation

So, by doing this operations step by step we can train versatile and more accurate llm model to give the factually correct informations and to mitigate hallucinations.

-------Approach---- Building the RAG Pipeline-------
•	Embeddings & Vectorization: Use a pre-trained model (like Hugging Face models) to convert text into embeddings. Store these embeddings in a vector database such as FAISS or Pinecone.
•	Retrieval: Implement a retrieval system that uses vector search to find the most relevant sections of the factsheet based on the user’s query. This can be done using FAISS or LangChain with its built-in vector stores.
•	Generation: Use open source model (e.g., HuggingFace or a fine-tuned LLM) to generate answers based on the retrieved documents. This model should also be capable of performing simple arithmetic or financial calculations.


