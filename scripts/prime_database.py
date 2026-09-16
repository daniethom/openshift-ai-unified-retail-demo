"""Prime the Milvus knowledge base from project JSON data files."""

from __future__ import annotations

import logging
import sys

from config.settings import settings
from rag.service import get_rag_service

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


def prime_database(recreate: bool = False) -> None:
    logger.info("Starting Milvus priming against %s", settings.milvus_uri)
    logger.info("Target collection: %s", settings.milvus_collection_name)

    service = get_rag_service()
    result = service.prime_from_data_files(recreate=recreate)
    inserted = result.get("inserted", 0)

    if inserted == 0:
        logger.error("No documents were inserted. Check data files and Milvus connectivity.")
        sys.exit(1)

    logger.info("Database priming complete. Inserted %s documents.", inserted)


if __name__ == "__main__":
    recreate_flag = "--recreate" in sys.argv
    prime_database(recreate=recreate_flag)
