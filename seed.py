"""Seed the Chroma collection with architecture decision records.

Run once before the app:

    uv run python seed.py
"""

from dotenv import load_dotenv

load_dotenv()
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

ARCHITECTURE_RECORDS = [
    Document(
        page_content=(
            "Customer-generated CSV exports are written to the exports S3 bucket "
            "under a per-tenant prefix and expire after 30 days via a lifecycle rule. "
            "Exports are never stored on application servers."
        ),
        metadata={"source": "ADR-001", "title": "CSV export storage"},
    ),
    Document(
        page_content=(
            "All services expose REST APIs with JSON payloads. gRPC is reserved for "
            "internal high-throughput calls between the ingestion and indexing services."
        ),
        metadata={"source": "ADR-002", "title": "API style"},
    ),
    Document(
        page_content=(
            "PostgreSQL is the system of record for transactional data. Each service "
            "owns its own schema and no service reads another service's tables directly."
        ),
        metadata={"source": "ADR-003", "title": "Primary database"},
    ),
    Document(
        page_content=(
            "Asynchronous work is dispatched through RabbitMQ. Messages must be "
            "idempotent because consumers use at-least-once delivery."
        ),
        metadata={"source": "ADR-004", "title": "Message broker"},
    ),
    Document(
        page_content=(
            "Authentication uses OAuth 2.0 with short-lived JWT access tokens (15 minutes) "
            "and refresh tokens stored in HTTP-only cookies."
        ),
        metadata={"source": "ADR-005", "title": "Authentication"},
    ),
    Document(
        page_content=(
            "Secrets are stored in HashiCorp Vault and injected at deploy time. "
            "Secrets must never be committed to the repository or baked into images."
        ),
        metadata={"source": "ADR-006", "title": "Secrets management"},
    ),
    Document(
        page_content=(
            "Structured JSON logs are shipped to Loki and traces to Tempo. Every request "
            "carries a correlation ID that is propagated to all downstream calls."
        ),
        metadata={"source": "ADR-007", "title": "Observability"},
    ),
    Document(
        page_content=(
            "Redis is used as a read-through cache with a default TTL of 5 minutes. "
            "Cache keys are namespaced by service and tenant."
        ),
        metadata={"source": "ADR-008", "title": "Caching"},
    ),
    Document(
        page_content=(
            "Uploaded files are scanned for malware before being accepted and are "
            "limited to 25 MB. Larger uploads must use the chunked upload endpoint."
        ),
        metadata={"source": "ADR-009", "title": "File uploads"},
    ),
    Document(
        page_content=(
            "Personal data is retained for 24 months after account closure and then "
            "hard-deleted. Backups containing personal data are rotated within 35 days."
        ),
        metadata={"source": "ADR-010", "title": "Data retention"},
    ),
]


def main() -> None:
    """Upsert the ADRs into Chroma, using the ADR id as the document id."""
    vector_store = Chroma(
        collection_name="architecture_records",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="./chroma_db",
    )
    vector_store.add_documents(
        ARCHITECTURE_RECORDS,
        ids=[doc.metadata["source"] for doc in ARCHITECTURE_RECORDS],
    )
    print(f"Seeded {len(ARCHITECTURE_RECORDS)} architecture records.")


if __name__ == "__main__":
    main()