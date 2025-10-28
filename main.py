import argparse
from src.pipelines.training_pipeline import run_training
# from src.pipelines.inference_pipeline import run_inference

def main():
    parser = argparse.ArgumentParser(description="Project Entry Point")

    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "infer"], 
        default="train",
        help="Which pipeline to run"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="If infer mode: model checkpoint path"
    )

    args = parser.parse_args()
    
    if args.mode == "train":
        print(f"Starting training using {args.config}")
        run_training(args.config)

    elif args.mode == "infer":
        if args.checkpoint is None:
            raise ValueError("Inference requires --checkpoint")
        print(f"Running inference using {args.config}, checkpoint={args.checkpoint}")
        run_inference(args.config, args.checkpoint)

if __name__ == "__main__":
    main()
