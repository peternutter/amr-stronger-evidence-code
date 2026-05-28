import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import torch
import wandb
import os
import shutil
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@hydra.main(
    version_base="1.1", config_path="../../../configs", config_name="train_model_trl.yaml"
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Initialize Weights & Biases if enabled
    use_wandb = bool(cfg.logging.use_wandb)
    if use_wandb:
        try:
            wandb.init(
                project=cfg.logging.project_name,
                name=cfg.logging.run_name,
                entity=cfg.logging.entity,
            )
        except Exception as e:
            print(
                f"Unexpected error initializing WandB: {e}. Disabling WandB logging for this run."
            )
            use_wandb = False

    df = pd.read_csv(cfg.data.path)
    if getattr(cfg.data, "sample", None): 
        df = df.sample(frac=cfg.data.sample if cfg.data.sample < 1 else None, 
                    n=int(cfg.data.sample) if cfg.data.sample >= 1 else None, 
                    random_state=42)

    if "query" not in df.columns or cfg.data.answer_column not in df.columns:
        raise ValueError(f"CSV must contain 'query' and '{cfg.data.answer_column}' columns")

    df["text"] = "[Human]: " + df["query"] + "\n[Assistant]: " + df[cfg.data.answer_column]
    dataset = Dataset.from_pandas(df[["text"]])

    model_name = cfg.model.name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Choose dtype and optionally use bitsandbytes quantization
    # If use_4bit is true create a BitsAndBytesConfig and pass quantization_config
    bnb_config = None
    if cfg.training.use_4bit:
        # compute dtype for quantized ops
        compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and cfg.training.use_bf16)
        else torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto",
        offload_folder=(
            cfg.training.offload_folder if cfg.training.use_offload else None
        ),
    )

    if cfg.training.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Attach LoRA if requested
    if cfg.training.use_lora:
        lora_cfg = LoraConfig(
            r=cfg.peft.r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=list(OmegaConf.to_container(cfg.peft.target_modules, resolve=True)),
            lora_dropout=cfg.peft.lora_dropout,
            bias=cfg.peft.bias,
            task_type=cfg.peft.task_type,
        )
        model = get_peft_model(model, lora_cfg)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable_params} of {total_params}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=None,
        args=SFTConfig(
            output_dir=cfg.training.output_dir,
            dataset_text_field="text",
            max_grad_norm=cfg.training.max_grad_norm if "max_grad_norm" in cfg.training else 1.0,
            num_train_epochs=cfg.training.epochs,
            per_device_train_batch_size=cfg.training.batch_size,
            learning_rate=cfg.training.learning_rate,
            max_length=cfg.training.max_length,
            save_strategy="epoch",
            logging_steps=cfg.logging.log_steps,
            report_to="wandb" if cfg.logging.use_wandb else "none",
            fp16=cfg.training.fp16,
            fp16_full_eval=cfg.training.fp16,
            packing=False,
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            optim=cfg.training.optim if "optim" in cfg.training else "adamw_torch",
        ),
    )

    trainer.train()

    # Save final model and tokenizer
    if cfg.training.use_lora:
        print("Saving adapter only in save_dir")
        model.save_pretrained(cfg.training.save_dir)
        tokenizer.save_pretrained(cfg.training.save_dir)
    else:
        trainer.model.save_pretrained(cfg.training.save_dir)
        trainer.tokenizer.save_pretrained(cfg.training.save_dir)

    # Optionally delete checkpoint directory
    if cfg.training.delete_checkpoints_after_final_save:
        if os.path.exists(cfg.training.output_dir):
            print(f"Deleting checkpoint directory {cfg.training.output_dir}")
            shutil.rmtree(cfg.training.output_dir)

    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
