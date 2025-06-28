import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import polars as pl
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_vector_db(db_path: str = "./chroma_db") -> chromadb.Client:
    """
    Initializes a local ChromaDB client.

    Args:
        db_path (str): The path where the ChromaDB should be stored.

    Returns:
        chromadb.Client: An initialized ChromaDB client.
    """
    try:
        # CORRECCIÓN: Pasa la configuración para permitir el reinicio
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(allow_reset=True)
        )
        logger.info(f"ChromaDB client initialized at {db_path}")
        return client
    except Exception as e:
        logger.error(f"Error initializing ChromaDB client: {e}")
        raise

def create_or_get_collection(client: chromadb.Client, collection_name: str) -> chromadb.Collection:
    """
    Creates a new collection or gets an existing one in ChromaDB.

    Args:
        client (chromadb.Client): The ChromaDB client.
        collection_name (str): The name of the collection.

    Returns:
        chromadb.Collection: The ChromaDB collection object.
    """
    try:
        collection = client.get_or_create_collection(name=collection_name)
        logger.info(f"Collection '{collection_name}' created or retrieved successfully.")
        return collection
    except Exception as e:
        logger.error(f"Error creating or getting collection '{collection_name}': {e}")
        raise

def load_embedding_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Loads a SentenceTransformer embedding model.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"Embedding model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model '{model_name}': {e}")
        raise

def generate_embeddings(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of texts using the given model.

    Args:
        model (SentenceTransformer): The embedding model.
        texts (list[str]): A list of texts to embed.

    Returns:
        list[list[float]]: A list of embeddings, where each embedding is a list of floats.
    """
    try:
        embeddings = model.encode(texts).tolist()
        logger.info(f"Generated embeddings for {len(texts)} documents.")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def add_documents_to_collection(
    collection: chromadb.Collection,
    documents: list[str],
    embeddings: list[list[float]],
    ids: list[str]
) -> None:
    """
    Adds documents, their embeddings, and IDs to a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The ChromaDB collection.
        documents (list[str]): The list of documents (texts).
        embeddings (list[list[float]]): The corresponding embeddings for the documents.
        ids (list[str]): Unique IDs for each document.
    """
    if not documents:
        logger.warning("Attempted to add an empty list of documents. Skipping.")
        return  # <-- Termina la ejecución aquí

    try:
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )
        logger.info(f"Added {len(documents)} documents to the collection.")
    except Exception as e:
        logger.error(f"Error adding documents to collection: {e}")
        # En un entorno de prueba, es mejor dejar que el error ocurra para detectarlo.
        # En producción, podrías manejarlo de otra forma.
        raise

def query_vector_db(
    collection: chromadb.Collection,
    query_embedding: list[list[float]],
    n_results: int = 2
) -> pl.DataFrame:
    """
    Queries the vector database for the most similar documents.

    Args:
        collection (chromadb.Collection): The ChromaDB collection to query.
        query_embedding (list[list[float]]): The embedding of the query text.
        n_results (int): The number of nearest results to retrieve.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the query results (documents, distances, ids).
    """
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'distances', 'metadatas']
        )
        # Convert results to Polars DataFrame
        if results and 'documents' in results and results['documents']:
            # Flatten the list of lists for documents, distances, and ids
            flat_documents = [doc for sublist in results['documents'] for doc in sublist]
            flat_distances = [dist for sublist in results['distances'] for dist in sublist]
            flat_ids = [idx for sublist in results['ids'] for idx in sublist]

            df = pl.DataFrame({
                "id": flat_ids,
                "document": flat_documents,
                "distance": flat_distances
            })
            logger.info(f"Query returned {len(flat_documents)} results.")
            return df
        else:
            logger.warning("No documents found for the given query.")
            return pl.DataFrame({"id": pl.Series([], dtype=pl.Utf8), "document": pl.Series([], dtype=pl.Utf8), "distance": pl.Series([], dtype=pl.Float64)})
    except Exception as e:
        logger.error(f"Error querying vector database: {e}")
        raise

def get_collection_data(collection: chromadb.Collection) -> pl.DataFrame:
    """
    Retrieves all documents, their IDs, and embeddings from a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The ChromaDB collection.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the IDs, documents, and embeddings.
    """
    try:
        data = collection.get(include=["embeddings", "documents"])
        if data and 'documents' in data and data['documents']:
            df = pl.DataFrame({
                "id": data['ids'],
                "document": data['documents'],
                "embedding": data['embeddings']
            })
            logger.info(f"Retrieved {len(data['documents'])} documents from collection.")
            return df
        else:
            logger.warning("No documents found in the collection.")
            return pl.DataFrame({"id": pl.Series([], dtype=pl.Utf8), "document": pl.Series([], dtype=pl.Utf8), "embedding": pl.Series([], dtype=pl.Object)})
    except Exception as e:
        logger.error(f"Error retrieving data from collection: {e}")
        raise
