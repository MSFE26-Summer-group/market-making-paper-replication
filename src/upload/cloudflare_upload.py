import os
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "data" / "rds_transformed.parquet"


def progress_callback(bytes_transferred: int) -> None:
    print(f"\rUploaded: {bytes_transferred / (1024*1024):.1f} MB", end="")


def main() -> None:
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )

    config = TransferConfig(
        multipart_threshold=1024 * 25,
        multipart_chunksize=1024 * 25,
        max_concurrency=4,
        use_threads=True,
    )

    s3.upload_file(
        str(DATA_FILE),
        os.environ["R2_BUCKET"],
        "raw/btc_usdt_lob_snapshot_2022-10-20.parquet",
        Config=config,
        Callback=progress_callback,
    )
    print("\ndone")


if __name__ == "__main__":
    main()
