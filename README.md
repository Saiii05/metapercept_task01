This project implements a simple Retrieval Augmented Generation (RAG) pipeline using ChromaDB as the vector store and sentence transformers for generating embeddings.

## Features

- Initializes a persistent ChromaDB vector database.
- Uses `all-MiniLM-L6-v2` sentence transformer for text embeddings.
- Preprocesses text by converting to lowercase and removing punctuation.
- Allows adding documents to the ChromaDB collection.
- Allows querying the collection to retrieve relevant documents based on semantic similarity.
- Includes a demonstration of the pipeline with sample data.

## Setup and Installation

1.  **Clone the repository (if applicable) or ensure `rag_pipeline.py` is in your desired directory.**

2.  **Install the required Python libraries:**
    Open your terminal and run:
    ```bash
    pip install chromadb sentence-transformers
    ```

## Running the Script

To run the RAG pipeline demonstration, execute the following command in your terminal from the directory containing `rag_pipeline.py`:

```bash
python rag_pipeline.py
```

This will:
- Create a `./chroma_db` directory in the same location as the script, which will store the persistent ChromaDB data.
- Add five sample documents to a collection named `my_rag_collection`.
- Query the collection with a sample question: "What is the tallest mountain?".
- Print the most relevant document found and its distance score.

## How it Works

1.  **Setup (`setup_chroma`)**: Initializes a `PersistentClient` for ChromaDB, storing data in the `./chroma_db` directory. It also sets up the `SentenceTransformerEmbeddingFunction` using the `all-MiniLM-L6-v2` model.
2.  **Preprocessing (`preprocess_text`)**: Converts input text to lowercase and removes all punctuation defined in Python's `string.punctuation`. This is applied to both documents before storage and queries before searching.
3.  **Collection Management (`get_or_create_collection`)**: Retrieves or creates a ChromaDB collection with the specified name and embedding function.
4.  **Document Indexing (`add_documents_to_collection`)**: Preprocesses the input documents and then adds them to the collection along with their unique IDs.
5.  **Querying (`query_collection`)**: Preprocesses the input query texts and then uses the collection's `query` method to find the `n_results` most similar documents.
6.  **Main (`main`)**: Orchestrates the pipeline by calling the above functions with sample data to demonstrate the workflow.

## Customization

-   **Embedding Model**: You can change the sentence transformer model in `setup_chroma()` by modifying the `model_name` parameter. Refer to the [Sentence Transformers documentation](https://www.sbert.net/docs/pretrained_models.html) for available models.
-   **Persistent Storage Path**: The path for ChromaDB storage can be changed in `setup_chroma()` by modifying the `path` parameter for `chromadb.PersistentClient`.
-   **Documents and Queries**: Modify the `documents`, `ids`, and `query_texts` variables in the `main()` function to use your own data.
-   **Preprocessing**: The `preprocess_text` function can be extended or modified to include other preprocessing steps like stemming, lemmatization, or stop word removal if needed.
