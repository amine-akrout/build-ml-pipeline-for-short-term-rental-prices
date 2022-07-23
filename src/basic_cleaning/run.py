#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    dataframe = pd.read_csv(artifact_local_path, index_col="id")
    min_price = args.min_price
    max_price = args.max_price
    idx = dataframe['price'].between(min_price, max_price)
    dataframe = dataframe[idx].copy()
    logger.info("Dataset price outliers removal outside range: %s-%s",
                 args.min_price, args.max_price)
    dataframe['last_review'] = pd.to_datetime(dataframe['last_review'])
    logger.info("Dataset last_review data type fix")

    idx = dataframe['longitude'].between(-74.25, -73.50) & dataframe['latitude'].between(40.5, 41.2)
    dataframe = dataframe[idx].copy()

    filename = args.output_artifact
    dataframe.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file(filename)
    logger.info("Cleaned dataset uploaded to wandb")
    run.log_artifact(artifact)
    artifact.wait()
    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help='the input artifact',
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help='the type for the output artifact',
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help='the type for the output artifact',
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help='a description for the output artifact',
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help='the minimum price limit',
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help='the maximum price limit',
        required=True
    )


    args = parser.parse_args()

    go(args)