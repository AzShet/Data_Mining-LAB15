# Advanced Text Mining with Vector Databases

This repository contains the final, fully-tested project for the Data Mining course at TECSUP. It serves as a deep dive into modern text mining techniques, moving beyond traditional keyword-based search to implement a robust semantic search system. The core of this project is the application of sentence embeddings and vector databases to find information based on conceptual meaning and context, rather than just lexical matches.

The project demonstrates the end-to-end lifecycle of a data science experiment: from initial implementation in a Jupyter Notebook, to refactoring core logic into reusable utility functions, and finally, to building a comprehensive and resilient automated testing suite using professional testing patterns.

<br>

## Table of Contents

- [Project Information](#project-information)
  - [Instructor](#instructor)
- [1. Project Overview](#1-project-overview)
- [2. Key Concepts Demonstrated](#2-key-concepts-demonstrated)
- [3. Project Structure](#3-project-structure)
- [4. Technologies Used](#4-technologies-used)
- [5. Setup and Installation](#5-setup-and-installation)
- [6. Usage](#6-usage)
  - [A. Running the Jupyter Notebook](#a-running-the-jupyter-notebook)
  - [B. Running the Automated Tests](#b-running-the-automated-tests)
  - [Testing Methodology](#testing-methodology)
- [7. Acknowledgements](#7-acknowledgements)
- [8. License](#8-license)

---

## Project Information

* **Student:** [César Diego Ruelas Flores](https://www.linkedin.com/in/diego-ruelas-flores/)
* **Program:** Big Data and Data Science
* **Institution:** [TECSUP](https://www.tecsup.edu.pe/)
* **Course:** Data Mining
* **Date:** June 28, 2025

### Instructor

> **[Luis Paraguay Arzapalo](https://www.linkedin.com/in/luisparaguay/)**
>
> Systems Engineer and Master in Information Technology Management from ESAN and La Salle, Spain. Specialist in Big Data, Business Intelligence, Machine Learning, Cloud, SQL, Data Modeling, and Agility.

---

## 1. Project Overview

The primary objective of this project is to build and compare two semantic search engines using different embedding models. We simulate a real-world scenario where we need to query a small knowledge base of personal information to find the most relevant individuals based on natural language questions.

This is achieved by:

1. **Vectorizing Text:** Transforming text documents into high-dimensional numerical vectors (embeddings) using state-of-the-art models from the `sentence-transformers` library.
2. **Indexing Vectors:** Storing these embeddings in `ChromaDB`, a specialized open-source vector database designed for efficient similarity searches.
3. **Querying by Meaning:** Converting a user's query into an embedding and using ChromaDB to retrieve the most semantically similar document vectors from the database based on cosine similarity.
4. **Comparing Models:** Demonstrating how the choice of embedding model (`all-MiniLM-L6-v2` vs. `multi-qa-MiniLM-L6-cos-v1`) can influence the relevance and ranking of search results, highlighting the importance of model selection for specific tasks (e.g., general vs. question-answering contexts).

---

## 2. Key Concepts Demonstrated

This project provides practical implementation of several cutting-edge concepts in NLP and data retrieval:

* **Sentence Embeddings:** We leverage pre-trained Transformer models to map sentences and paragraphs into a dense vector space. In this space, texts with similar meanings are located closer to each other, a property that forms the foundation of semantic search.

* **Vector Databases:** We use **ChromaDB** to manage our embeddings. Unlike traditional relational databases that query based on exact matches in structured data, a vector database excels at finding the "nearest neighbors" to a given query vector in a high-dimensional space, making it ideal for similarity-based tasks.

* **Semantic Search vs. Lexical Search:**
  * **Lexical Search (Traditional):** Finds documents containing exact keywords (e.g., searching for "data science" would miss a document that only says "machine learning and statistics").
  * **Semantic Search (This Project):** Finds documents based on conceptual similarity. A query like "¿Quién estudia ciencia de datos?" can successfully match the document "Mi nombre es Jean Pierre Ruelas, soy estudiante de Ciencia de Datos y me especializo en Machine Learning," because the system understands the meaning behind the words, not just the words themselves.

* **Test-Driven Development (TDD) Principles:** The project includes a robust test suite built with **Pytest**. We apply advanced testing patterns, such as using fixtures for setup/teardown and `monkeypatch` to isolate tests from the file system, ensuring fast, reliable, and deterministic test runs.

---

## 3. Project Structure

The repository is organized to separate concerns, following best practices for data science projects.

```
.
├── .venv/                   # Virtual environment directory
├── Data_Mining-LAB15/       # Main project folder
│   ├── notebooks/
│   │   └── LAB15-RUELAS.ipynb # The main Jupyter Notebook for experimentation and presentation
│   ├── src/
│   │   └── utils.py           # Reusable Python functions for database and model interactions
│   └── tests/
│       └── test_utils.py      # Pytest unit tests for all functions in utils.py
├── .gitignore               # Standard git ignore file
├── README.md                # This file
└── requirements.txt         # Project dependencies
```

* **`notebooks/LAB15-RUELAS.ipynb`**: The interactive notebook where the initial exploration, comparison of models, and final demonstration are performed.
* **`src/utils.py`**: A Python module containing all the core logic, refactored from the notebook. This includes functions for initializing the database, creating collections, loading models, generating embeddings, and querying the database. This promotes code reuse and maintainability.
* **`tests/test_utils.py`**: The automated test suite. It contains unit tests for every function in `utils.py`, ensuring that each component of our system works as expected.

---

## 4. Technologies Used

* **Python 3.12+**
* **Jupyter Lab**: For interactive development and demonstration.
* **ChromaDB**: The open-source vector database for storing and querying embeddings.
* **Sentence-Transformers**: For loading state-of-the-art models and generating text embeddings.
* **Polars**: A high-performance DataFrame library used for handling and displaying query results.
* **Pytest**: The framework used for writing and running the automated test suite.

---

## 5. Setup and Installation

To run this project locally, follow these steps.

**1. Clone the Repository**

```bash
git clone <your-repository-url>
cd <your-repository-directory>
```

**2. Create and Activate a Virtual Environment**

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

**3. Install Dependencies**

Install all required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

*(Note: A `requirements.txt` file would typically contain `jupyterlab`, `chromadb`, `sentence-transformers`, `polars`, and `pytest`)*

---

## 6. Usage

### A. Running the Jupyter Notebook

To explore the project's main demonstration and see the comparison between the embedding models, run the Jupyter Notebook.

1. Launch Jupyter Lab from your terminal:
   ```bash
   jupyter lab
   ```
2. Navigate to the `notebooks/` directory in the Jupyter interface.
3. Open `LAB15-RUELAS.ipynb`.
4. You can run all cells sequentially to see the entire process, from database creation to the final comparison of query results.

### B. Running the Automated Tests

To verify the correctness and robustness of all the underlying functions, run the test suite using Pytest.

1. From the root directory of the project in your terminal, simply run:

   ```bash
   pytest
   ```

2. You should see the following output, indicating that all 8 tests have passed successfully:

   ```
   ============================= test session starts =============================
   ...
   collected 8 items

   tests/test_utils.py::test_initialize_vector_db PASSED                  [ 12%]
   tests/test_utils.py::test_create_or_get_collection PASSED              [ 25%]
   tests/test_utils.py::test_load_embedding_model PASSED                  [ 37%]
   tests/test_utils.py::test_generate_embeddings PASSED                   [ 50%]
   tests/test_utils.py::test_query_vector_db PASSED                       [ 62%]
   tests/test_utils.py::test_get_collection_data PASSED                   [ 75%]
   tests/test_utils.py::test_query_vector_db_no_results PASSED            [ 87%]
   tests/test_utils.py::test_add_documents_empty_lists PASSED             [100%]

   ============================== 8 passed in ...s ===============================
   ```

### Testing Methodology

The test suite in `tests/test_utils.py` was carefully crafted to be both comprehensive and efficient. The key testing strategy employed is the use of **`pytest-monkeypatch`** to ensure **test isolation**.

For the tests, the on-disk `chromadb.PersistentClient` is swapped out at runtime for an in-memory `chromadb.EphemeralClient`. This is the professional standard for testing database-dependent code because:

* **It eliminates file I/O:** Tests do not create, modify, or delete files on disk, which completely prevents `PermissionError` issues and other file system flakiness.
* **It's extremely fast:** In-memory operations are orders of magnitude faster than disk operations.
* **It guarantees isolation:** Each test function gets its own pristine, in-memory database, ensuring that tests do not interfere with one another.

This advanced testing pattern allows for a fast, reliable, and deterministic verification of the entire application logic without changing a single line of the production code in `src/utils.py`.

---

## 7. Acknowledgements

I would like to extend my sincere gratitude to our instructor, **Luis Paraguay Arzapalo**, for his exceptional guidance, expertise, and mentorship throughout the Data Mining course. His deep knowledge in Big Data, Machine Learning, and modern data technologies was instrumental in shaping the concepts and successful implementation of this project. His practical approach and encouragement to adopt professional software engineering practices, such as automated testing, have been invaluable to my learning journey.

Thank you for fostering a challenging and supportive learning environment.

---

## 8. License

This project is licensed under the MIT License. See the `LICENSE` file for details.