import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from typing import Optional
from tqdm import tqdm

from datasets import load_dataset


@hydra.main(
    version_base="1.1", config_path="configs", config_name="inference_freeform"
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load data: CSV or HF dataset
    df: Optional[pd.DataFrame] = None
    use_hf = getattr(cfg.data, "hf", None) is not None
    print(f"Using HuggingFace dataset: {use_hf}")

    if not use_hf:
        # CSV path provided
        df = pd.read_csv(cfg.data.path)
        if getattr(cfg.data, "sample", None):
            df = df.sample(
                frac=cfg.data.sample if cfg.data.sample < 1 else None,
                n=int(cfg.data.sample) if cfg.data.sample >= 1 else None,
                random_state=42,
            )
        question_col = getattr(cfg.data, "question_column", "question")
        if question_col not in df.columns:
            raise ValueError(f"CSV must contain a '{question_col}' column")
        df = df.rename(columns={question_col: "query"})
    else:
        hf_name = getattr(cfg.data, "name", "gsm8k")
        hf_subset = getattr(cfg.data.hf, "hf_subset", "main")
        hf_split = getattr(cfg.data.hf, "hf_split", "test")

        ds = load_dataset(hf_name, name=hf_subset, split=hf_split)
        # Optional sub-sampling for HF datasets
        sample = getattr(cfg.data, "sample", None)
        if sample:
            total_size = len(ds)
            if 0 < float(sample) < 1:
                n_samples = int(total_size * float(sample))
            else:
                n_samples = min(int(sample), total_size)
            ds = ds.shuffle(seed=42).select(range(n_samples))
        
        # Build dataframe with a 'query' column and optional gold answer
        data_dict = {"query": ds["question"]}
        data_dict["answer"] = ds["answer"]
        df = pd.DataFrame(data_dict)

    print(f"Running evaluation on {len(df)} datapoints...")

    queries = df["query"].astype(str).tolist()

    model_name = cfg.model.name
    checkpoint = (
        cfg.model.checkpoint if getattr(cfg.model, "checkpoint", None) else model_name
    )
    adapter = getattr(cfg.model, "adapter", None)
    use_peft = bool(getattr(cfg.model, "use_peft", False))
    use_4bit = bool(getattr(cfg.model, "use_4bit", False))
    
    # If adapter is null/None, disable PEFT regardless of config
    if adapter is None:
        use_peft = False

    print(f"Loading tokenizer from {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()

    # Quantization config if needed
    bnb_config = None
    if use_4bit:
        compute_dtype = torch.float16 if use_cuda else torch.float32
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("Using 4-bit quantized model loading")

    # Load model
    dtype = torch.bfloat16 if use_cuda else torch.float32
    print(f"Loading model from {checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto" if use_cuda else None,
        torch_dtype=dtype,
        quantization_config=bnb_config,
    )

    # Load adapter if requested
    if use_peft or adapter:
        adapter_path = adapter
        print(f"Loading PEFT adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

        if bool(getattr(cfg.model, "merge_adapter", False)):
            print("Merging adapter into base model for faster inference")
            try:
                model = model.merge_and_unload()
            except Exception as e:
                print(f"Could not merge adapter: {e}")

    model.eval()

    gen_kwargs = dict(
        max_new_tokens=int(cfg.inference.max_new_tokens),
        pad_token_id=tokenizer.eos_token_id,
        temperature=float(getattr(cfg.inference, "temperature", 0.7)),
        top_p=float(getattr(cfg.inference, "top_p", 0.9)),
    )

    batch_size = int(cfg.inference.batch_size)
    outputs = []

    def generate_batch(batch_queries):
        prompts = ["[Human]: " + q + "\n[Assistant]: " for q in batch_queries]
        encoding = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(cfg.inference.max_input_length),
        )
        input_ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

        results = []
        for prompt, full in zip(prompts, decoded):
            full_str = full.strip()
            if full_str.startswith(prompt):
                answer = full_str[len(prompt) :].strip()
            else:
                answer = full_str.split("\n", 1)[-1].strip()
            results.append(answer)
        return results

    for i in tqdm(range(0, len(queries), batch_size), desc="Running inference"):
        batch = queries[i : i + batch_size]
        try:
            batch_out = generate_batch(batch)
        except Exception as e:
            print(f"Error on batch starting at {i}: {e}")
            batch_out = ["" for _ in batch]
        outputs.extend(batch_out)

    df["model_answer"] = outputs[: len(df)]

    if adapter:
        model_id = adapter.rstrip("/").split("finetuned_on_")[1].split("/")[0] + "_" + adapter.rstrip("/").split("/")[-2]
    else:
        model_id = "base_model"
    
    output_filename = f"{cfg.data.name}_{model_id}_results.csv"

    out_dir = (
        cfg.data.output_dir
        if hasattr(cfg.data, "output_dir") and cfg.data.output_dir
        else os.getcwd()
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_filename)
    df.to_csv(out_path, index=False)
    print(f"Saved inference results to {out_path}")


if __name__ == "__main__":
    main()
