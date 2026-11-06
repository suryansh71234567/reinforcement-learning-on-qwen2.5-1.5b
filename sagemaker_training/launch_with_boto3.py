#!/usr/bin/env python3
"""
Launch SageMaker GRPO training job using pure boto3 (no sagemaker SDK needed).
Uses YOUR AWS credentials for account 421669311560.

Run from Bash:
  python launch_with_boto3.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import boto3

# ── Credentials (from environment variables) ──────────────────────────────
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
REGION = os.environ.get("AWS_REGION", "us-east-1")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_USERNAME = os.environ.get("HF_USERNAME", "Suryansh7123")
HF_REPO = os.environ.get("HF_REPO", f"{HF_USERNAME}/llama3.2-3b-grpo-metamath")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "grpo-llama3b-metamath")

ROLE_ARN = os.environ.get("SM_ROLE_ARN", "arn:aws:iam::421669311560:role/SageMakerGRPOTrainingRole")
S3_BUCKET = os.environ.get("S3_BUCKET", "grpo-suryansh-2026")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.g5.xlarge")

# Validate
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    print("ERROR: Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars")
    sys.exit(1)
if not HF_TOKEN:
    print("WARNING: HF_TOKEN not set — cannot push checkpoints to Hub")
if not WANDB_API_KEY:
    print("WARNING: WANDB_API_KEY not set — WandB disabled")

# ── AWS session ────────────────────────────────────────────────────────────
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION,
)
sm_client = session.client("sagemaker")
sts_client = session.client("sts")

# Verify account
identity = sts_client.get_caller_identity()
print(f"[INFO] Account: {identity['Account']}")
print(f"[INFO] User: {identity['Arn']}")

if identity["Account"] != "421669311560":
    print(f"WARNING: Expected account 421669311560, got {identity['Account']}")

# ── SageMaker PyTorch 2.1.0 image (us-east-1) ─────────────────────────────
IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
)

# ── S3 paths ───────────────────────────────────────────────────────────────
checkpoint_s3_uri = f"s3://{S3_BUCKET}/checkpoints"
output_s3_uri = f"s3://{S3_BUCKET}/output"
code_s3_uri = f"s3://{S3_BUCKET}/code"

# ── Upload training code to S3 ────────────────────────────────────────────
print("[INFO] Uploading training code to S3...")
s3_client = session.client("s3")

# Upload the training script and requirements
for filepath in ["sagemaker_grpo_train.py", "requirements.txt"]:
    if os.path.exists(filepath):
        s3_key = f"code/{filepath}"
        s3_client.upload_file(filepath, S3_BUCKET, s3_key)
        print(f"  Uploaded {filepath} -> s3://{S3_BUCKET}/{s3_key}")
    else:
        print(f"  WARNING: {filepath} not found, skipping")

# ── Training job config ────────────────────────────────────────────────────
job_name = f"grpo-llama3b-a10-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

hyperparameters = {
    "model-name": "meta-llama/Llama-3.2-3B-Instruct",
    "max-seq-length": "2048",
    "lora-r": "16",
    "lora-alpha": "32",
    "lora-dropout": "0.05",
    "group-size": "8",
    "batch-size": "8",
    "grad-accum": "2",
    "max-steps": "1000",
    "max-new-tokens": "500",
    "temperature": "0.8",
    "beta": "0.03",
    "max-grad-norm": "0.1",
    "learning-rate": "5e-7",
    "warmup-steps": "50",
    "lr-scheduler-type": "cosine",
    "length-threshold": "300",
    "w-correctness": "0.80",
    "w-length": "0.20",
    "dataset-name": "meta-math/MetaMathQA",
    "heldout-every-n-steps": "100",
    "heldout-n-samples": "200",
    "checkpoint-every-n-steps": "200",
    "sample-log-every-n-steps": "50",
    "hf-repo": HF_REPO,
    "seed": "42",
}

environment = {
    "HF_TOKEN": HF_TOKEN,
    "HF_USERNAME": HF_USERNAME,
    "HF_REPO": HF_REPO,
    "WANDB_PROJECT": WANDB_PROJECT,
    "WANDB_RUN_NAME": job_name,
    "WANDB_API_KEY": WANDB_API_KEY,
}

print(f"[INFO] Training job: {job_name}")
print(f"[INFO] Instance: {INSTANCE_TYPE} (A10 24GB)")
print(f"[INFO] Spot: enabled (up to 90% savings)")
print(f"[INFO] Checkpoint S3: {checkpoint_s3_uri}")

# ── Create training job ────────────────────────────────────────────────────
try:
    response = sm_client.create_training_job(
        TrainingJobName=job_name,
        HyperParameters=hyperparameters,
        AlgorithmSpecification={
            "TrainingImage": IMAGE_URI,
            "TrainingInputMode": "File",
        },
        RoleArn=ROLE_ARN,
        OutputDataConfig={
            "S3OutputPath": output_s3_uri,
        },
        ResourceConfig={
            "InstanceType": INSTANCE_TYPE,
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 86400,  # 24 hours
        },
        EnableManagedSpotTraining=True,
        CheckpointConfig={
            "S3Uri": checkpoint_s3_uri,
            "LocalPath": "/opt/ml/checkpoints",
        },
        Environment=environment,
        Tags=[
            {"Key": "Project", "Value": "GRPO-Llama3B-MetaMath"},
            {"Key": "Hackathon", "Value": "SyntaxError2026"},
        ],
    )

    print()
    print("=" * 60)
    print(f"  🚀 TRAINING JOB LAUNCHED")
    print(f"  Job name     : {job_name}")
    print(f"  Status       : Starting...")
    print(f"  Instance     : {INSTANCE_TYPE} (A10 24GB)")
    print(f"  Spot         : Yes (managed)")
    print(f"  Checkpoints  : {checkpoint_s3_uri}")
    print(f"  Console      : https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/{job_name}")
    print("=" * 60)
    print()
    print("Monitor with:")
    print(f"  aws sagemaker describe-training-job --training-job-name {job_name} --query TrainingJobStatus")
    print()
    print("View logs:")
    print(f"  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-names-prefix {job_name} --follow")

except Exception as e:
    print(f"ERROR: Failed to create training job: {e}")
    sys.exit(1)
