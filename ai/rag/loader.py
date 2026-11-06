from pathlib import Path
import pandas as pd

from langchain_core.documents import Document


def load_text_file(file_path: Path):
    """
    Load a single TXT knowledge-base file.
    """

    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "file_type": "policy",
            },
        )
    ]


def load_csv_file(file_path: Path):
    """
    Load a CSV knowledge-base file.

    Each row becomes one LangChain Document.
    """

    data = pd.read_csv(file_path)

    documents = []

    for index, row in data.iterrows():

        # Convert every non-null field into readable text.
        fields = []

        for column, value in row.items():

            if pd.isna(value):
                continue

            # Skip reference fields for product rows to avoid
            # creating chunks containing only a generic URL.
            if (
                file_path.name == "product_information.csv"
                and column in {"source_name", "source_url"}
            ):
                continue

            value = str(value).strip()

            if value:
                fields.append(f"{column}: {value}")

        if not fields:
            continue

        text = "\n".join(fields)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "file_type": "csv",
                    "row": int(index),
                },
            )
        )

    return documents


def load_knowledge_base(base_path="data/knowledge_base"):
    """
    Load all RAG knowledge-base files.

    Supported:
    - TXT policy files
    - CSV files
    """

    base_path = Path(base_path)

    if not base_path.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {base_path}"
        )

    documents = []

    # Load policy TXT files
    policy_path = base_path / "policies"

    if policy_path.exists():

        for file_path in sorted(policy_path.glob("*.txt")):
            docs = load_text_file(file_path)
            documents.extend(docs)

    # Load CSV knowledge

    for file_path in sorted(base_path.glob("*.csv")):

        docs = load_csv_file(file_path)
        documents.extend(docs)

    return documents

if __name__ == "__main__":
    documents = load_knowledge_base()

    print(f"Loaded {len(documents)} documents.")

    for doc in documents[:5]:
        print("\n---")
        print("Source:", doc.metadata["source"])
        print(doc.page_content[:300])