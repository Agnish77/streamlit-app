import streamlit as st
from transformers import pipeline
import torch
import os
import boto3

bucket_name = "agnishpaul"
local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'

device = 0 if torch.cuda.is_available() else -1

s3 = boto3.client('s3')

def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file)


st.title("Machine Learning Model Deployment at the Server!!!")

if st.button("Download Model"):
    with st.spinner("Downloading..."):
        download_dir(local_path, s3_prefix)
        st.success("Model downloaded!")

# 👉 Only load model **after it exists**
if os.path.exists(local_path):
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
