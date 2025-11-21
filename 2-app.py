import streamlit as st
from transformers import pipeline
import torch
import os
import boto3

st.title("Machine Learning Model Deployment at the Server!!!")

# Load AWS creds directly (do NOT set os.environ)
AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION = st.secrets["AWS_DEFAULT_REGION"]

bucket_name = "agnishpaul"

# Correct S3 prefix of your model folder
s3_prefix = "ml-models/tinybert-sentiment-analysis/"

# Local folder where model is stored
local_path = "tinybert-sentiment-analysis"

device = 0 if torch.cuda.is_available() else -1

# Correct boto3 client (uses credentials directly)
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION,
)


def download_dir(local_path, s3_prefix):
    """Download an entire S3 directory to a local path."""
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")

    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" in result:
            for key in result["Contents"]:
                s3_key = key["Key"]

                # Skip folder keys
                if s3_key.endswith("/"):
                    continue

                rel_path = os.path.relpath(s3_key, s3_prefix)
                local_file = os.path.join(local_path, rel_path)

                # Ensure subdirectory exists
                os.makedirs(os.path.dirname(local_file), exist_ok=True)

                s3.download_file(bucket_name, s3_key, local_file)


# Button to download model
if st.button("Download Model"):
    with st.spinner("Downloading model files from S3..."):
        download_dir(local_path, s3_prefix)
    st.success("Model downloaded successfully!")


# Load pipeline ONLY if config.json exists
config_path = os.path.join(local_path, "config.json")

if os.path.exists(config_path):
    st.success("Model found! Ready to use.")

    classifier = pipeline(
        "text-classification",
        model=local_path,
        tokenizer=local_path,
        device=device,
    )

    text = st.text_area("Enter Your Review")

    if st.button("Predict"):
        st.write(classifier(text))

else:
    st.warning("Model not downloaded yet. Click 'Download Model' first.")

