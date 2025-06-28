import pytest
import os
import chromadb
import shutil
import time 
import polars as pl
from sentence_transformers import SentenceTransformer
from chromadb.api.client import Client as ChromaClient
from chromadb.api.models.Collection import Collection

# Importar las funciones que vamos a probar desde el archivo utils
from src.utils import (
    initialize_vector_db,
    create_or_get_collection,
    load_embedding_model,
    generate_embeddings,
    add_documents_to_collection,
    query_vector_db,
    get_collection_data
)

# --- Fixtures para las pruebas (configuración que se reutiliza) ---

@pytest.fixture(scope="function")
def setup_teardown_db(monkeypatch):
    """
    Usa monkeypatch para reemplazar el cliente de disco por uno en memoria,
    manejando los argumentos de forma inteligente.
    """
    monkeypatch.setattr(
        chromadb,
        "PersistentClient",
        lambda path, settings: chromadb.EphemeralClient(settings=settings)
    )

    # Ahora esta llamada es segura. Nuestra lambda maneja los argumentos.
    client = initialize_vector_db("dummy_path_for_test")
    yield client

    # La limpieza en memoria es simple y ahora funcionará sin errores.
    client.reset()

@pytest.fixture(scope="module")
def embedding_model():
    """
    Carga el modelo de embedding una sola vez para todas las pruebas.
    """
    return load_embedding_model('all-MiniLM-L6-v2')

# --- Casos de Prueba ---

def test_initialize_vector_db(setup_teardown_db):
    """
    Prueba que el cliente de la base de datos se inicializa correctamente.
    """
    client = setup_teardown_db
    assert isinstance(client, ChromaClient)
    assert os.path.exists("./test_chroma_db")

def test_create_or_get_collection(setup_teardown_db):
    """
    Prueba que se puede crear y obtener una colección.
    """
    client = setup_teardown_db
    collection_name = "test_collection"
    collection = create_or_get_collection(client, collection_name)
    assert isinstance(collection, Collection)
    assert collection.name == collection_name

def test_load_embedding_model():
    """
    Prueba que el modelo de embedding se carga correctamente.
    """
    model = load_embedding_model('all-MiniLM-L6-v2')
    assert isinstance(model, SentenceTransformer)

def test_generate_embeddings(embedding_model):
    """
    Prueba que se generan embeddings correctamente.
    """
    texts = ["hello world", "test sentence"]
    embeddings = generate_embeddings(embedding_model, texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)

def test_query_vector_db(setup_teardown_db, embedding_model):
    """
    Prueba que las consultas a la base de datos devuelven resultados.
    """
    client = setup_teardown_db
    collection = create_or_get_collection(client, "query_test_collection")

    docs = ["apple is a fruit", "banana is a fruit", "car is a vehicle"]
    ids = ["d1", "d2", "d3"]
    embeddings = generate_embeddings(embedding_model, docs)
    add_documents_to_collection(collection, docs, embeddings, ids)

    query_embedding = generate_embeddings(embedding_model, ["What is a fruit?"])
    results_df = query_vector_db(collection, query_embedding, n_results=2)

    assert isinstance(results_df, pl.DataFrame)
    assert len(results_df) == 2

def test_get_collection_data(setup_teardown_db, embedding_model):
    """
    Prueba que se pueden obtener todos los datos de una colección.
    """
    client = setup_teardown_db
    collection = create_or_get_collection(client, "get_data_test_collection")

    docs = ["item A", "item B"]
    ids = ["id_A", "id_B"]
    embeddings = generate_embeddings(embedding_model, docs)
    add_documents_to_collection(collection, docs, embeddings, ids)

    df = get_collection_data(collection)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 2
    assert "item A" in df["document"].to_list()

def test_query_vector_db_no_results(setup_teardown_db, embedding_model):
    """
    Prueba que una consulta sin resultados devuelve un DataFrame vacío.
    """
    client = setup_teardown_db
    collection = create_or_get_collection(client, "empty_query_collection")
    query_embedding = generate_embeddings(embedding_model, ["non-existent item"])
    results_df = query_vector_db(collection, query_embedding, n_results=2)
    assert results_df.is_empty()

def test_add_documents_empty_lists(setup_teardown_db):
    """
    Prueba que al intentar añadir listas vacías, la operación no hace nada y no falla.
    """
    client = setup_teardown_db
    collection = create_or_get_collection(client, "empty_add_collection")

    # Esta llamada ahora no debería hacer nada gracias a la guarda en utils.py
    add_documents_to_collection(collection, [], [], [])

    df = get_collection_data(collection)
    assert df.is_empty()
