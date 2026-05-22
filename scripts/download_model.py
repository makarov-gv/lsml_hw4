import argparse
from pathlib import Path

from minio import Minio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--access-key", required=True)
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = Minio(
        args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        secure=False,
    )
    client.fget_object(args.bucket, args.object_name, str(output_path))
    print(f"Downloaded s3://{args.bucket}/{args.object_name} to {output_path}")


if __name__ == "__main__":
    main()
