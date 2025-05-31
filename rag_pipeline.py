# Mini RAG Pipeline with ChromaDB
#
# Required libraries:
# pip install chromadb sentence-transformers

import chromadb
from chromadb.utils import embedding_functions
import string

# Global variable for the embedding function
DEFAULT_EMBEDDING_FUNCTION = None

def setup_chroma():
    """Initializes and returns a ChromaDB client and a sentence transformer embedding function."""
    global DEFAULT_EMBEDDING_FUNCTION
    # Initialize ChromaDB client (persistent storage)
    client = chromadb.PersistentClient(path="./chroma_db")

    # Use a pre-built sentence transformer embedding function
    DEFAULT_EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    return client, DEFAULT_EMBEDDING_FUNCTION

def preprocess_text(text: str) -> str:
    """Converts text to lowercase and removes punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def get_or_create_collection(client, collection_name, embedding_function):
    """
    Retrieves an existing ChromaDB collection or creates a new one if it doesn't exist.

    Args:
        client: The ChromaDB client instance.
        collection_name (str): The name of the collection.
        embedding_function: The embedding function to use for the collection.

    Returns:
        chromadb.api.models.Collection.Collection: The retrieved or created collection.
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    return collection

def add_documents_to_collection(collection, documents, ids):
    """
    Adds preprocessed documents to the specified ChromaDB collection.

    Args:
        collection: The ChromaDB collection instance.
        documents (list[str]): A list of documents (strings) to add.
        ids (list[str]): A list of unique IDs for the documents.
    """
    processed_documents = [preprocess_text(doc) for doc in documents]
    collection.add(
        documents=processed_documents,
        ids=ids
    )

def query_collection(collection, query_texts, n_results=2):
    """
    Preprocesses query texts and then queries the ChromaDB collection,
    returning the most relevant documents.

    Args:
        collection: The ChromaDB collection instance.
        query_texts (list[str]): A list of query texts (strings).
        n_results (int): The number of results to return for each query.

    Returns:
        dict: The query results from ChromaDB.
    """
    processed_query_texts = [preprocess_text(query) for query in query_texts]
    results = collection.query(
        query_texts=processed_query_texts,
        n_results=n_results
    )
    return results

def main():
    """
    Main function to demonstrate the RAG pipeline:
    1. Sets up ChromaDB client and embedding function.
    2. Gets or creates a collection.
    3. Adds sample documents to the collection.
    4. Queries the collection with a sample query.
    5. Prints the results.
    """
    # Setup ChromaDB
    client, embedding_function = setup_chroma()

    # Get or create a collection
    collection_name = "my_rag_collection"
    collection = get_or_create_collection(client, collection_name, embedding_function)

    # Sample documents to add to the collection
    documents = [
        "The Eiffel Tower is located in Paris.",
        "The Great Wall of China is one of the seven wonders of the world.",
        "The Mona Lisa was painted by Leonardo da Vinci.",
        "The Amazon rainforest is the largest tropical rainforest in the world.",
        "Mount Everest is the highest mountain above sea level."
    ]
    ids = [f"doc{i}" for i in range(len(documents))]

    # Add documents to the collection
    add_documents_to_collection(collection, documents, ids)
    print(f"Added {len(documents)} documents to the collection '{collection_name}'.")

    # Query the collection
    query_texts = ["What is the tallest mountain?"]
    results = query_collection(collection, query_texts, n_results=1)

    print("\nQuery Results:")
    if results and results.get('documents'):
        for i, doc in enumerate(results['documents'][0]):
            print(f"  Result {i+1}: {doc}")
            if results.get('distances') and results['distances'][0]:
                 print(f"  Distance: {results['distances'][0][i]:.4f}")
    else:
        print("No results found or error in query.")

if __name__ == "__main__":
    main()
