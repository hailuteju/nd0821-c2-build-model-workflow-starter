#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting
the result to a new artifact
"""
import argparse
import logging
import os
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f"Download artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Drop samples with null values for `room_type`")
    df.dropna(subset=["room_type"], inplace=True)

    logger.info("Drop outliers")
    idx = (df['price'].between(args.min_price, args.max_price)
           & df['longitude'].between(-74.25, -73.50)
           & df['latitude'].between(40.5, 41.2))

    df = df[idx].copy()

    logger.info("Convert last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Save data frame to a csv file")
    filename = "clean_sample.csv"
    df.to_csv(filename, index=False)

    logger.info("Load the clean data to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact to create",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price",
        required=True
    )

    args = parser.parse_args()

    go(args)
