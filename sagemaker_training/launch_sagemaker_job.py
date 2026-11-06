#!/usr/bin/env python3
"""
Launch a SageMaker training job for GRPO fine-tuning with Unsloth + TRL
on an A10 GPU (ml.g5.xlarge) using spot instances.

Usage:
    python launch_sagemaker_job.py <IAM_ROLE_ARN> --region us-east-1

Environment variables required:
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN (optional)
    HF_TOKEN, HF_USERNAME, HF_REPO, WANDB_API_KEY

Hyperparameters match grpo_train.py exactly:
  meta-llama/Llama-3.2-3B-Instruct, LoRA r=16, group_size=8, batch=8,
  max_steps=1000, lr=5e-7, beta=0.03, temperature=0.8, bf16
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import boto3
from sagemaker.pytorch import PyTorch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a SageMaker GRPO training job on ml.g5.xlarge (A10)."
    )
    parser.add_argument(
        "role",
        help="IAM execution role ARN for SageMaker.",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS region (default: us-east-1). us-east-1 has ml.g5.xlarge spot quota.",
    )
    parser.add_argument(
        "--instance-type",
        default="ml.g5.xlarge",
        help="SageMaker instance type (default: ml.g5.xlarge, 1x A10 24GB).",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of instances (default: 1).",
    )
    parser.add_argument(
        "--max-run",
        type=int,
        default=86400,
        help="Max training time in seconds (default: 86400 = 24h).",
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        default=172800,
        help="Max wait time for spot instances in seconds (default: 172800 = 48h).",
    )
    parser.add_argument(
        "--checkpoint-s3-uri",
        default="s3://grpo-suryansh-2026/checkpoints",
        help="S3 URI for checkpoints.",
    )
    parser.add_argument(
        "--no-spot",
        action="store_true",
        help="Disable spot instances (use on-demand).",
    )
    args = parser.parse_args()

    # --- Validate required environment variables -----------------------------
    required_env = ["HF_TOKEN", "HF_USERNAME", "HF_REPO", "WANDB_API_KEY"]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        print(f"ERROR: missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    # --- AWS session --------------------------------------------------------
    session = boto3.Session(region_name=args.region)
    sm_client = session.client("sagemaker")
    account_id = session.client("sts").get_caller_identity()["Account"]
    region = args.region

    print(f"[INFO] AWS region   : {region}")
    print(f"[INFO] AWS account  : {account_id}")
    print(f"[INFO] IAM role     : {args.role}")
    print(f"[INFO] Instance     : {args.instance_type} x {args.instance_count}")
    print(f"[INFO] Spot enabled : {not args.no_spot}")

    # --- Build the PyTorch estimator ----------------------------------------
    # Hyperparameters EXACTLY match grpo_train.py config
    estimator = PyTorch(
        entry_point="sagemaker_grpo_train.py",
        source_dir="./",
        role=args.role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        framework_version="2.1.0",
        py_version="py311",
        max_run=args.max_run,
        checkpoint_s3_uri=args.checkpoint_s3_uri,
        use_spot_instances=not args.no_spot,
        max_wait=args.max_wait if not args.no_spot else None,
        dependencies=["requirements.txt"],
        environment={
            "HF_TOKEN": os.environ["HF_TOKEN"],
            "HF_USERNAME": os.environ["HF_USERNAME"],
            "HF_REPO": os.environ["HF_REPO"],
            "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "grpo-llama3b-metamath"),
            "WANDB_RUN_NAME": os.environ.get(
                "WANDB_RUN_NAME",
                f"grpo-a10-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            ),
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        },
        hyperparameters={
            # Model (MUST match grpo_train.py)
            "model_name": "meta-llama/Llama-3.2-3B-Instruct",
            "max_seq_length": 2048,
            # LoRA
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            # GRPO
            "group_size": 8,
            "per_device_batch": 8,
            "grad_accum_steps": 2,
            "max_steps": 1000,
            "max_new_tokens": 500,
            "temperature": 0.8,
            "beta": 0.03,
            "max_grad_norm": 0.1,
            # Training
            "learning_rate": 5e-7,
            "warmup_steps": 50,
            "lr_scheduler_type": "cosine",
            # Reward
            "w_correctness": 0.80,
            "w_length": 0.20,
            "length_threshold": 300,
            # Data
            "dataset_name": "meta-math/MetaMathQA",
            "heldout_every_n_steps": 100,
            "heldout_n_samples": 200,
            "checkpoint_every_n_steps": 200,
            "sample_log_every_n_steps": 50,
            # Output
            "hf_repo": os.environ["HF_REPO"],
            # Seed
            "seed": 42,
        },
    )

    # --- Launch the job -----------------------------------------------------
    job_name = (
        f"grpo-llama3b-a10-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    print(f"[INFO] Training job : {job_name}")

    estimator.fit(
        job_name=job_name,
        wait=False,
    )

    # --- Report status ------------------------------------------------------
    desc = sm_client.describe_training_job(TrainingJobName=job_name)
    status = desc["TrainingJobStatus"]
    secondary = desc.get("SecondaryStatus", "N/A")

    print()
    print("=" * 60)
    print(f"  Training job name  : {job_name}")
    print(f"  Status             : {status}")
    print(f"  Secondary status   : {secondary}")
    print(f"  Instance           : {args.instance_type}")
    print(f"  Spot               : {not args.no_spot}")
    print(f"  Checkpoint S3 URI  : {args.checkpoint_s3_uri}")
    print(
        f"  SageMaker console  : "
        f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
