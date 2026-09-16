"""Download the Granite model into a local directory or PVC mount."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def _env(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return value.strip() if value else default


def download_model(
    repo_id: str,
    target_dir: Path,
    hf_token: str | None = None,
    revision: str | None = None,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install model dependencies with: pip install -e '.[model]'"
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", repo_id, target_dir)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=hf_token or None,
        revision=revision or None,
    )

    marker = target_dir / ".download_complete"
    marker.write_text(repo_id, encoding="utf-8")
    logger.info("Model download complete.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Granite model weights.")
    parser.add_argument(
        "--repo",
        default=_env("MODEL_HF_REPO", "ibm-granite/granite-3.0-2b-instruct"),
        help="Hugging Face model repository",
    )
    parser.add_argument(
        "--output",
        default=_env("MODEL_LOCAL_DIR", "./models/granite-3b"),
        help="Directory to store model files",
    )
    parser.add_argument(
        "--revision",
        default=_env("MODEL_HF_REVISION") or None,
        help="Optional model revision or branch",
    )
    args = parser.parse_args()

    token = _env("HF_TOKEN") or None
    target = Path(args.output).resolve()

    try:
        download_model(
            repo_id=args.repo,
            target_dir=target,
            hf_token=token,
            revision=args.revision,
        )
    except Exception as exc:
        logger.error("Model download failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
