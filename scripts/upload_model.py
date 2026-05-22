import argparse
from pathlib import Path

from minio import Minio

from ml_utils import save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--access-key", required=True)
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--upload-info-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {input_path}")

    client = Minio(
        args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        secure=False,
    )
    if not client.bucket_exists(args.bucket):
        client.make_bucket(args.bucket)

    result = client.fput_object(
        args.bucket,
        args.object_name,
        str(input_path),
        content_type="application/octet-stream",
    )
    upload_info = {
        "bucket": args.bucket,
        "object": args.object_name,
        "etag": result.etag,
        "version_id": result.version_id,
        "source_file": str(input_path),
        "source_size_bytes": input_path.stat().st_size,
        "console_url": "http://localhost:19001",
    }
    save_json(upload_info, args.upload_info_path)
    print(f"Uploaded {input_path} to s3://{args.bucket}/{args.object_name}")


if __name__ == "__main__":
    main()
