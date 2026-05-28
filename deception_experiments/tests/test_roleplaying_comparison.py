"""
Compare roleplaying (offpolicy_train) implementations between our code and Apollo's.

Tests that both implementations produce:
1. The same formatted prompts (after tokenizer.apply_chat_template)
2. The same detection masks (which tokens we train on)

These are prefill datasets - no generation needed.
"""

import sys
from pathlib import Path

import pytest
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
APOLLO_ROOT = PROJECT_ROOT / "deception-detection"

# Ensure imports work
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(APOLLO_ROOT))

from deception_detection.data.roleplaying import RolePlayingDataset  # noqa: E402
from deception_detection.tokenized_data import TokenizedDataset  # noqa: E402
from deception_detection.types import Message  # noqa: E402
from deception_detection.utils import preprocess_dialogue  # noqa: E402
from src.data.roleplaying import RolePlayingDataModule  # noqa: E402
from src.utils.mask_utils import collate_fn_with_prefill  # noqa: E402


class TestRoleplayingDataFormat:
    """Test that both implementations handle the YAML data the same way."""

    @pytest.fixture
    def sample_scenario(self):
        """Get a sample scenario from the dataset."""
        yaml_path = APOLLO_ROOT / "data/roleplaying/dataset.yaml"
        if not yaml_path.exists():
            pytest.skip("roleplaying dataset.yaml not found")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Return first scenario
        scenario = data[0]
        # Strip quotation marks as both implementations do
        return {
            "scenario": scenario["scenario"],
            "question": scenario["question"],
            "answer_prefix": scenario.get("answer_prefix", "").strip('"'),
            "honest_completion": scenario["honest_completion"].strip('"'),
            "deceptive_completion": scenario["deceptive_completion"].strip('"'),
        }

    def test_offpolicy_train_dialogue_structure(self, llama_tokenizer, sample_scenario):
        """Both implementations should create the same dialogue structure for offpolicy_train."""
        scenario = sample_scenario

        # ----- Apollo's implementation -----
        # For offpolicy_train, Apollo creates:
        # [Message(system, scenario, False), Message(user, question, False), Message(assistant, completion, True)]
        apollo_deceptive_dialogue = [
            Message("system", scenario["scenario"], False),
            Message("user", scenario["question"], False),
            Message("assistant", scenario["deceptive_completion"], True),
        ]

        apollo_honest_dialogue = [
            Message("system", scenario["scenario"], False),
            Message("user", scenario["question"], False),
            Message("assistant", scenario["honest_completion"], True),
        ]

        # ----- Our implementation -----
        # We create the same structure
        our_deceptive_conversation = [
            {"role": "system", "content": scenario["scenario"]},
            {"role": "user", "content": scenario["question"]},
            {"role": "assistant", "content": scenario["deceptive_completion"]},
        ]

        our_honest_conversation = [
            {"role": "system", "content": scenario["scenario"]},
            {"role": "user", "content": scenario["question"]},
            {"role": "assistant", "content": scenario["honest_completion"]},
        ]

        # Format both using tokenizer
        apollo_deceptive_processed = preprocess_dialogue(apollo_deceptive_dialogue)
        apollo_deceptive_formatted = llama_tokenizer.apply_chat_template(
            apollo_deceptive_processed,
            tokenize=False,
            add_generation_prompt=False,
        )

        our_deceptive_formatted = llama_tokenizer.apply_chat_template(
            our_deceptive_conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        # They should be identical
        assert our_deceptive_formatted == apollo_deceptive_formatted, (
            f"Deceptive dialogue format mismatch:\n"
            f"Ours:\n{our_deceptive_formatted}\n\n"
            f"Apollo:\n{apollo_deceptive_formatted}"
        )

        # Check honest too
        apollo_honest_processed = preprocess_dialogue(apollo_honest_dialogue)
        apollo_honest_formatted = llama_tokenizer.apply_chat_template(
            apollo_honest_processed,
            tokenize=False,
            add_generation_prompt=False,
        )

        our_honest_formatted = llama_tokenizer.apply_chat_template(
            our_honest_conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        assert our_honest_formatted == apollo_honest_formatted, (
            f"Honest dialogue format mismatch:\n"
            f"Ours:\n{our_honest_formatted}\n\n"
            f"Apollo:\n{apollo_honest_formatted}"
        )


class TestRoleplayingDetectionMask:
    """Test that detection masks match between implementations."""

    @pytest.fixture
    def sample_scenario(self):
        """Get a sample scenario from the dataset."""
        yaml_path = APOLLO_ROOT / "data/roleplaying/dataset.yaml"
        if not yaml_path.exists():
            pytest.skip("roleplaying dataset.yaml not found")

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        scenario = data[0]
        return {
            "scenario": scenario["scenario"],
            "question": scenario["question"],
            "answer_prefix": scenario.get("answer_prefix", "").strip('"'),
            "honest_completion": scenario["honest_completion"].strip('"'),
            "deceptive_completion": scenario["deceptive_completion"].strip('"'),
        }

    def test_detection_mask_tokens_match_no_padding(self, llama_tokenizer, sample_scenario):
        """Both implementations should mask the same tokens (the completion content)."""
        scenario = sample_scenario

        # ----- Apollo's implementation -----
        apollo_dialogue = [
            Message("system", scenario["scenario"], False),
            Message("user", scenario["question"], False),
            Message("assistant", scenario["deceptive_completion"], True),  # detect=True
        ]

        # Apollo uses 0 padding for roleplaying
        padding_config = {
            "llama": {"left": 0, "right": 0},
            "gemma": {"left": 0, "right": 0},
            "mistral": {"left": 0, "right": 0},
        }

        apollo_tokenized = TokenizedDataset.from_dialogue_list(
            [apollo_dialogue],
            llama_tokenizer,
            padding=padding_config,
        )

        apollo_tokens = apollo_tokenized.tokens[0]
        apollo_mask = apollo_tokenized.detection_mask[0]

        # ----- Our implementation -----
        our_conversation = [
            {"role": "system", "content": scenario["scenario"]},
            {"role": "user", "content": scenario["question"]},
            {"role": "assistant", "content": scenario["deceptive_completion"]},
        ]

        batch = [{"conversation": our_conversation}]
        our_result = collate_fn_with_prefill(batch, llama_tokenizer, apply_padding=False)

        our_tokens = our_result["input_ids"][0]
        our_mask = our_result["detection_mask"][0]

        # ----- Compare masked content -----
        apollo_masked_tokens = apollo_tokens[apollo_mask]
        apollo_masked_text = llama_tokenizer.decode(apollo_masked_tokens, skip_special_tokens=True)

        our_masked_tokens = our_tokens[our_mask]
        our_masked_text = llama_tokenizer.decode(our_masked_tokens, skip_special_tokens=True)

        assert our_masked_text.strip() == apollo_masked_text.strip(), (
            f"Masked content mismatch:\n" f"Ours: '{our_masked_text}'\n" f"Apollo: '{apollo_masked_text}'"
        )

        # Both should mask the completion content
        completion = scenario["deceptive_completion"]
        assert completion.strip() in apollo_masked_text or apollo_masked_text.strip() in completion
        assert completion.strip() in our_masked_text or our_masked_text.strip() in completion

    def test_detection_mask_count_match(self, llama_tokenizer, sample_scenario):
        """Both implementations should have the same number of masked tokens."""
        scenario = sample_scenario

        # ----- Apollo's implementation -----
        apollo_dialogue = [
            Message("system", scenario["scenario"], False),
            Message("user", scenario["question"], False),
            Message("assistant", scenario["deceptive_completion"], True),
        ]

        padding_config = {
            "llama": {"left": 0, "right": 0},
            "gemma": {"left": 0, "right": 0},
            "mistral": {"left": 0, "right": 0},
        }

        apollo_tokenized = TokenizedDataset.from_dialogue_list(
            [apollo_dialogue],
            llama_tokenizer,
            padding=padding_config,
        )

        apollo_mask_count = apollo_tokenized.detection_mask[0].sum().item()

        # ----- Our implementation -----
        our_conversation = [
            {"role": "system", "content": scenario["scenario"]},
            {"role": "user", "content": scenario["question"]},
            {"role": "assistant", "content": scenario["deceptive_completion"]},
        ]

        batch = [{"conversation": our_conversation}]
        our_result = collate_fn_with_prefill(batch, llama_tokenizer, apply_padding=False)

        our_mask_count = our_result["detection_mask"][0].sum().item()

        assert (
            our_mask_count == apollo_mask_count
        ), f"Mask token count mismatch: ours={our_mask_count}, apollo={apollo_mask_count}"

    def test_token_ids_match(self, llama_tokenizer, sample_scenario):
        """Both implementations should produce identical token sequences."""
        scenario = sample_scenario

        # ----- Apollo's implementation -----
        apollo_dialogue = [
            Message("system", scenario["scenario"], False),
            Message("user", scenario["question"], False),
            Message("assistant", scenario["deceptive_completion"], True),
        ]

        padding_config = {
            "llama": {"left": 0, "right": 0},
            "gemma": {"left": 0, "right": 0},
            "mistral": {"left": 0, "right": 0},
        }

        apollo_tokenized = TokenizedDataset.from_dialogue_list(
            [apollo_dialogue],
            llama_tokenizer,
            padding=padding_config,
        )

        apollo_tokens = apollo_tokenized.tokens[0]

        # ----- Our implementation -----
        our_conversation = [
            {"role": "system", "content": scenario["scenario"]},
            {"role": "user", "content": scenario["question"]},
            {"role": "assistant", "content": scenario["deceptive_completion"]},
        ]

        batch = [{"conversation": our_conversation}]
        our_result = collate_fn_with_prefill(batch, llama_tokenizer, apply_padding=False)

        our_tokens = our_result["input_ids"][0]

        # The tokens should be identical
        assert len(our_tokens) == len(
            apollo_tokens
        ), f"Token length mismatch: ours={len(our_tokens)}, apollo={len(apollo_tokens)}"

        # Compare token by token
        mismatches = []
        for i, (our_tok, apollo_tok) in enumerate(zip(our_tokens.tolist(), apollo_tokens.tolist(), strict=False)):
            if our_tok != apollo_tok:
                our_decoded = llama_tokenizer.decode([our_tok])
                apollo_decoded = llama_tokenizer.decode([apollo_tok])
                mismatches.append(
                    f"  pos {i}: ours={our_tok} ({our_decoded!r}), " f"apollo={apollo_tok} ({apollo_decoded!r})"
                )

        assert len(mismatches) == 0, "Token mismatches:\n" + "\n".join(mismatches[:10])


class TestRoleplayingPaddingBehavior:
    """Test that Apollo's padding settings are correctly applied."""

    def test_roleplaying_uses_zero_padding(self):
        """Verify Apollo uses zero padding for roleplaying dataset."""
        # Apollo's RolePlayingDataset has padding = {"llama": {"left": 0, "right": 0}, ...}
        assert RolePlayingDataset.padding["llama"] == {"left": 0, "right": 0}
        assert RolePlayingDataset.padding["gemma"] == {"left": 0, "right": 0}
        assert RolePlayingDataset.padding["mistral"] == {"left": 0, "right": 0}

    def test_our_implementation_default_no_padding(self):
        """Verify our implementation defaults to no padding for roleplaying."""
        # RolePlayingDataModule.use_apollo_padding should be False
        assert hasattr(RolePlayingDataModule, "use_apollo_padding")
        assert not RolePlayingDataModule.use_apollo_padding


class TestRoleplayingEndToEnd:
    """End-to-end comparison of both implementations."""

    @pytest.mark.slow
    def test_full_dataset_comparison(self, llama_tokenizer, roleplaying_data_dir):
        """Compare full dataset processing between implementations."""
        # Check if data exists
        yaml_path = APOLLO_ROOT / "data/roleplaying/dataset.yaml"
        if not yaml_path.exists():
            pytest.skip("roleplaying dataset.yaml not found")

        # Copy data to our data dir
        import shutil

        dest_path = roleplaying_data_dir / "dataset.yaml"
        if not dest_path.exists():
            shutil.copy(yaml_path, dest_path)

        # ----- Apollo's implementation -----
        # Disable shuffle to compare in original YAML order
        apollo_dataset = RolePlayingDataset(variant="offpolicy_train", shuffle_upon_init=False)

        # ----- Our implementation -----
        our_datamodule = RolePlayingDataModule(
            data_dir=str(roleplaying_data_dir),
            safe_name="roleplaying-offpolicy_train",
            prompt_variant="offpolicy_train",
            max_samples=None,
        )
        our_datamodule.tokenizer = llama_tokenizer
        our_datamodule.setup("fit")

        # Apollo creates pairs: deceptive, honest for each scenario
        # Our implementation does the same
        assert len(our_datamodule.dataset) == len(
            apollo_dataset
        ), f"Dataset size mismatch: ours={len(our_datamodule.dataset)}, apollo={len(apollo_dataset)}"

        # Compare several samples
        num_to_compare = min(20, len(our_datamodule.dataset))
        padding_config = {
            "llama": {"left": 0, "right": 0},
            "gemma": {"left": 0, "right": 0},
            "mistral": {"left": 0, "right": 0},
        }

        mismatches = []
        for i in range(num_to_compare):
            our_sample = our_datamodule.dataset[i]
            apollo_dialogue = apollo_dataset.dialogues[i]

            # Get Apollo's masked content
            apollo_tokenized = TokenizedDataset.from_dialogue_list(
                [apollo_dialogue],
                llama_tokenizer,
                padding=padding_config,
            )
            apollo_masked_text = llama_tokenizer.decode(
                apollo_tokenized.tokens[0][apollo_tokenized.detection_mask[0]],
                skip_special_tokens=True,
            )

            # Get our masked content
            batch = [{"conversation": our_sample["conversation"]}]
            our_result = collate_fn_with_prefill(batch, llama_tokenizer, apply_padding=False)
            our_masked_text = llama_tokenizer.decode(
                our_result["input_ids"][0][our_result["detection_mask"][0]],
                skip_special_tokens=True,
            )

            if our_masked_text.strip() != apollo_masked_text.strip():
                mismatches.append(
                    f"Sample {i}:\n" f"  Ours: {our_masked_text[:100]!r}\n" f"  Apollo: {apollo_masked_text[:100]!r}"
                )

        assert len(mismatches) == 0, f"Found {len(mismatches)} mismatches:\n" + "\n".join(mismatches[:5])

        print(f"✓ Compared {num_to_compare} samples successfully")

    @pytest.mark.slow
    def test_labels_match(self, llama_tokenizer, roleplaying_data_dir):
        """Verify labels match between implementations."""
        yaml_path = APOLLO_ROOT / "data/roleplaying/dataset.yaml"
        if not yaml_path.exists():
            pytest.skip("roleplaying dataset.yaml not found")

        import shutil

        dest_path = roleplaying_data_dir / "dataset.yaml"
        if not dest_path.exists():
            shutil.copy(yaml_path, dest_path)

        # Import label types
        from deception_detection.types import Label as ApolloLabel
        from src.utils.types import Label as OurLabel

        # Disable shuffle to compare in original YAML order
        apollo_dataset = RolePlayingDataset(variant="offpolicy_train", shuffle_upon_init=False)

        our_datamodule = RolePlayingDataModule(
            data_dir=str(roleplaying_data_dir),
            safe_name="roleplaying-offpolicy_train",
            prompt_variant="offpolicy_train",
        )
        our_datamodule.tokenizer = llama_tokenizer
        our_datamodule.setup("fit")

        # Apollo: deceptive, honest pairs
        # Ours: deceptive, honest pairs
        num_to_compare = min(20, len(our_datamodule.dataset))

        for i in range(num_to_compare):
            our_label = our_datamodule.dataset[i]["label"]
            apollo_label = apollo_dataset.labels[i]

            # Convert to comparable format
            # Our Label uses integers (0=HONEST, 1=DECEPTIVE)
            # Apollo's Label is an Enum
            our_is_deceptive = our_label == OurLabel.DECEPTIVE  # our_label == 1
            apollo_is_deceptive = apollo_label == ApolloLabel.DECEPTIVE

            assert our_is_deceptive == apollo_is_deceptive, (
                f"Label mismatch at index {i}: ours={our_label} (deceptive={our_is_deceptive}), "
                f"apollo={apollo_label} (deceptive={apollo_is_deceptive})"
            )

        print(f"✓ Labels match for {num_to_compare} samples")
