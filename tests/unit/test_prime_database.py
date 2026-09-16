"""Smoke tests for Milvus priming script."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_prime_database_calls_rag_service():
    mock_service = MagicMock()
    mock_service.prime_from_data_files.return_value = {"inserted": 3}

    with patch("scripts.prime_database.get_rag_service", return_value=mock_service):
        from scripts.prime_database import prime_database

        prime_database(recreate=True)

    mock_service.prime_from_data_files.assert_called_once_with(recreate=True)
