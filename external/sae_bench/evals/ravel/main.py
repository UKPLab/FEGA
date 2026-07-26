import argparse
import gc
import os
import random
import time
from dataclasses import asdict
from datetime import datetime

import sae_bench.evals.ravel.intervention as intervention
import sae_bench.evals.ravel.mdbm as mdbm
import sae_bench.sae_bench_utils.activation_collection as activation_collection
import sae_bench.sae_bench_utils.general_utils as general_utils
import torch
from sae_bench.evals.ravel.eval_config import RAVELEvalConfig
from sae_bench.evals.ravel.eval_output import (
    EVAL_TYPE_ID_RAVEL,
    RAVELEvalOutput,
    RAVELMetricCategories,
    RAVELMetricResults,
)
from sae_bench.evals.ravel.generation import custom_left_padding
from sae_bench.evals.ravel.instance import (
    RAVELFilteredDataset,
    RAVELInstance,
    get_instance_name,
)
from sae_bench.evals.ravel.intervention import get_prompt_pairs
from sae_bench.sae_bench_utils import (
    get_eval_uuid,
    get_sae_bench_version,
    get_sae_lens_version,
)
from sae_bench.sae_bench_utils.sae_selection_utils import (
    get_saes_from_regex,
)
from sae_lens import SAE
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
)

# For our initial experiments, we only used transformer_lens and the shorter LLM names
# For RAVEL, we use HF transformers, which requires the full model name
# So, it's the shorter name: full name map
LLM_NAME_MAP = {
    "gpt2": "gpt2",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "pythia-70m-deduped": "EleutherAI/pythia-70m-deduped",
    "pythia-160m-deduped": "EleutherAI/pythia-160m-deduped",
}


def _save_mdbm_checkpoint(trained_mdbm: torch.nn.Module, path: str) -> None:
    """Save MDBM/MDAS tensors without pickling model, tokenizer, or SAE objects."""
    state_dict = {
        key: value.detach().cpu()
        for key, value in trained_mdbm.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    binary_mask = getattr(trained_mdbm, "binary_mask", None)
    if binary_mask is None:
        raise ValueError("Trained MDBM object has no binary_mask to save.")
    checkpoint = {
        "checkpoint_type": "ravel_mdbm_tensor_checkpoint",
        "class_name": trained_mdbm.__class__.__name__,
        "binary_mask": binary_mask.detach().cpu(),
        "state_dict": state_dict,
        "layer_intervened": getattr(trained_mdbm, "layer_intervened", None),
        "batch_size": getattr(trained_mdbm, "batch_size", None),
        "temperature": getattr(trained_mdbm, "temperature", None),
    }
    torch.save(checkpoint, path)


def _mdbm_result_dir(mdbm_dir: str, sae_result_path: str) -> str:
    """Key MDBM checkpoints by result filename, independent of output location."""
    return os.path.join(mdbm_dir, os.path.basename(sae_result_path))


# original create_dataloaders function
def create_dataloaders(
    cause_base_prompts,
    cause_source_prompts,
    iso_base_prompts,
    iso_source_prompts,
    model: PreTrainedModel,
    eval_config: RAVELEvalConfig,
    train_test_split: float,
):
    """
    Create train and validation dataloaders from prompt pairs.

    Args:
        cause_base_prompts: List of base prompts
        cause_source_prompts: List of source prompts
        model: The model (used for device information)
        eval_config: Configuration for evaluation
        train_test_split: Ratio of data to use for training (default: 0.5)

    Returns:
        train_loader: Dataloader for training
        val_loader: Dataloader for validation
    """
    # NOTE: Pay very close attention to the order of the arguments here and the difference between cause and iso
    # This determines the labels that are used for cause and iso
    formatted_cause_pairs = []
    for base, source in zip(cause_base_prompts, cause_source_prompts):
        formatted_cause_pairs.append(
            (
                base.input_ids,
                source.input_ids,
                base.attention_mask,
                source.attention_mask,
                base.final_entity_token_pos,
                source.final_entity_token_pos,
                base.first_generated_token_id,
                source.first_generated_token_id,
                base.text,
                source.attribute_label,  # NOTE: We want to change the label to source for cause
            )
        )

    formatted_iso_pairs = []
    for base, source in zip(iso_base_prompts, iso_source_prompts):
        formatted_iso_pairs.append(
            (
                base.input_ids,
                source.input_ids,
                base.attention_mask,
                source.attention_mask,
                base.final_entity_token_pos,
                source.final_entity_token_pos,
                base.first_generated_token_id,
                base.first_generated_token_id,  # NOTE: We want the label to remain as base for iso
                base.text,
                base.attribute_label,
            )
        )

    all_formatted_pairs = formatted_cause_pairs + formatted_iso_pairs
    random.seed(eval_config.random_seed)
    random.shuffle(all_formatted_pairs)

    # Split into train and validation sets
    total_pairs = len(all_formatted_pairs)
    train_size = int(total_pairs * train_test_split)

    train_pairs = all_formatted_pairs[:train_size]
    val_pairs = all_formatted_pairs[train_size:]

    print(
        f"Created {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs"
    )

    # Create dataloaders
    train_loader = create_dataloader_from_pairs(train_pairs, model, eval_config)
    val_loader = create_dataloader_from_pairs(val_pairs, model, eval_config)

    return train_loader, val_loader


# original create_dataloaders_from_pairs function
def create_dataloader_from_pairs(formatted_pairs, model, eval_config):
    """
    Create a dataloader from formatted prompt pairs.

    Args:
        formatted_pairs: List of formatted prompt pairs
        model: The model (used for device information)
        eval_config: Configuration for evaluation

    Returns:
        dataloader: List of batched data
    """
    dataloader = []
    num_batches = len(formatted_pairs) // eval_config.llm_batch_size

    tokenizer = AutoTokenizer.from_pretrained(eval_config.model_name)

    max_base_len = max(len(pair[0]) for pair in formatted_pairs)
    max_source_len = max(len(pair[1]) for pair in formatted_pairs)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * eval_config.llm_batch_size
        batch_end = batch_start + eval_config.llm_batch_size
        batch_data = formatted_pairs[batch_start:batch_end]

        base_tokens_BL = []
        source_tokens_BL = []
        base_attn_mask_BL = []
        source_attn_mask_BL = []
        base_pos_B = []
        source_pos_B = []
        base_pred_B = []
        source_pred_B = []
        base_text_str = []
        base_label_str = []

        for (
            base_tokens_L,
            source_tokens_L,
            base_attn_mask_L,
            source_attn_mask_L,
            base_pos,
            source_pos,
            base_pred,
            source_pred,
            base_text,
            base_label,
        ) in batch_data:
            base_tokens_BL.append(base_tokens_L)
            source_tokens_BL.append(source_tokens_L)
            base_attn_mask_BL.append(base_attn_mask_L)
            source_attn_mask_BL.append(source_attn_mask_L)
            base_pos_B.append(base_pos)
            source_pos_B.append(source_pos)
            base_pred_B.append(base_pred)
            source_pred_B.append(source_pred)
            base_text_str.append(base_text)
            base_label_str.append(base_label)

        base_tokens_BL, base_attn_mask_BL = custom_left_padding(
            tokenizer, base_tokens_BL, max_base_len
        )
        source_tokens_BL, source_attn_mask_BL = custom_left_padding(
            tokenizer, source_tokens_BL, max_source_len
        )

        base_tokens_BL = base_tokens_BL.to(model.device)
        base_attn_mask_BL = base_attn_mask_BL.to(model.device)
        source_tokens_BL = source_tokens_BL.to(model.device)
        source_attn_mask_BL = source_attn_mask_BL.to(model.device)

        base_pos_B = torch.tensor(base_pos_B).to(model.device)
        source_pos_B = torch.tensor(source_pos_B).to(model.device)
        base_pred_B = torch.tensor(base_pred_B).to(model.device)
        source_pred_B = torch.tensor(source_pred_B).to(model.device)

        base_encoding_BL = BatchEncoding(
            {
                "input_ids": base_tokens_BL,
                "attention_mask": base_attn_mask_BL,
            }
        )
        source_encoding_BL = BatchEncoding(
            {
                "input_ids": source_tokens_BL,
                "attention_mask": source_attn_mask_BL,
            }
        )

        dataloader.append(
            (
                base_encoding_BL,
                source_encoding_BL,
                base_pos_B,
                source_pos_B,
                base_pred_B,
                source_pred_B,
                base_text_str,
                base_label_str,
            )
        )

    print(f"Created dataloader with {len(dataloader)} batches")
    return dataloader


# Modified create_dataloaders function returning extra information for different attributes
def create_dataloaders_extended(
    cause_base_prompts,
    cause_source_prompts,
    iso_base_prompts,
    iso_source_prompts,
    model: PreTrainedModel,
    eval_config: RAVELEvalConfig,
    train_test_split: float,
):
    """
    Create train and validation dataloaders from prompt pairs.

    Args:
        cause_base_prompts: List of base prompts
        cause_source_prompts: List of source prompts
        model: The model (used for device information)
        eval_config: Configuration for evaluation
        train_test_split: Ratio of data to use for training (default: 0.5)

    Returns:
        train_loader: Dataloader for training
        val_loader: Dataloader for validation
    """
    # NOTE: Pay very close attention to the order of the arguments here and the difference between cause and iso
    # This determines the labels that are used for cause and iso
    formatted_cause_pairs = []
    for base, source in zip(cause_base_prompts, cause_source_prompts):
        formatted_cause_pairs.append(
            (
                source.text,  # source prompt
                base.entity_label,  # entity label of the base prompt (ex: "Paris")
                source.entity_label,  # entity label of the base prompt (ex: "Tokyo")
                base.attribute_type,  # base attribute type (ex: "Country")
                source.attribute_type,  # source attribute type (ex: "Country")
                base.attribute_label,  # base attribute label (ex: "France")
                source.attribute_label,  # source attribute label (ex: "Japan")
                base.input_ids,
                source.input_ids,
                base.attention_mask,
                source.attention_mask,
                base.final_entity_token_pos,  # final token pos for entity in base prompt
                source.final_entity_token_pos,
                base.first_generated_token_id,
                source.first_generated_token_id,
                base.text,  # base prompt
                source.attribute_label,  # NOTE: We want to change the label to source for cause
            )
        )

    formatted_iso_pairs = []
    for base, source in zip(iso_base_prompts, iso_source_prompts):
        formatted_iso_pairs.append(
            (
                source.text,  # source prompt
                base.entity_label,  # entity label of the base prompt (ex: "Paris")
                source.entity_label,  # entity label of the base prompt (ex: "Tokyo")
                base.attribute_type,  # base attribute type (ex: "Country")
                source.attribute_type,  # source attribute type (ex: "Country")
                base.attribute_label,  # base attribute label (ex: "France")
                source.attribute_label,  # source attribute label (ex: "Japan")
                base.input_ids,
                source.input_ids,
                base.attention_mask,
                source.attention_mask,
                base.final_entity_token_pos,  # final token pos for entity in base prompt
                source.final_entity_token_pos,
                base.first_generated_token_id,
                base.first_generated_token_id,
                base.text,  # base prompt
                base.attribute_label,  # NOTE: We want the label to remain as base for iso
            )
        )

    all_formatted_pairs = formatted_cause_pairs + formatted_iso_pairs
    random.seed(eval_config.random_seed)
    random.shuffle(all_formatted_pairs)

    # Split into train and validation sets
    total_pairs = len(all_formatted_pairs)
    train_size = int(total_pairs * train_test_split)

    train_pairs = all_formatted_pairs[:train_size]
    val_pairs = all_formatted_pairs[train_size:]

    print(
        f"Created {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs"
    )

    # Create dataloaders
    train_loader = create_dataloader_from_pairs_extended(
        train_pairs, model, eval_config
    )
    val_loader = create_dataloader_from_pairs_extended(val_pairs, model, eval_config)

    return train_loader, val_loader


# Modified create_dataloader_from_pairs function returning extra information for different attributes
def create_dataloader_from_pairs_extended(formatted_pairs, model, eval_config):
    """
    Create a dataloader from formatted prompt pairs.

    Args:
        formatted_pairs: List of formatted prompt pairs
        model: The model (used for device information)
        eval_config: Configuration for evaluation

    Returns:
        dataloader: List of batched data
    """
    dataloader = []
    num_batches = len(formatted_pairs) // eval_config.llm_batch_size

    tokenizer = AutoTokenizer.from_pretrained(eval_config.model_name)

    # create_dataloaders_extended() prepended 7 metadata fields before the original token fields,
    # so the input_ids are now at index 7 and 8 instead of 0 and 1
    max_base_len = max(len(pair[7]) for pair in formatted_pairs)
    max_source_len = max(len(pair[8]) for pair in formatted_pairs)

    for batch_idx in range(num_batches):
        batch_start = batch_idx * eval_config.llm_batch_size
        batch_end = batch_start + eval_config.llm_batch_size
        batch_data = formatted_pairs[batch_start:batch_end]

        source_text_str = []
        base_entity_labels = []
        source_entity_labels = []
        base_attr_types = []
        source_attr_types = []
        base_attr_labels = []
        source_attr_labels = []
        base_tokens_BL = []
        source_tokens_BL = []
        base_attn_mask_BL = []
        source_attn_mask_BL = []
        base_pos_B = []
        source_pos_B = []
        base_pred_B = []
        source_pred_B = []
        base_text_str = []
        base_label_str = []

        for (
            source_text,
            base_entity_label,
            source_entity_label,
            base_attribute_type,
            source_attribute_type,
            base_attribute_label,
            source_attribute_label,
            base_tokens_L,
            source_tokens_L,
            base_attn_mask_L,
            source_attn_mask_L,
            base_pos,
            source_pos,
            base_pred,
            source_pred,
            base_text,
            base_label,
        ) in batch_data:
            source_text_str.append(source_text)
            base_entity_labels.append(base_entity_label)
            source_entity_labels.append(source_entity_label)
            base_attr_types.append(base_attribute_type)
            source_attr_types.append(source_attribute_type)
            base_attr_labels.append(base_attribute_label)
            source_attr_labels.append(source_attribute_label)
            base_tokens_BL.append(base_tokens_L)
            source_tokens_BL.append(source_tokens_L)
            base_attn_mask_BL.append(base_attn_mask_L)
            source_attn_mask_BL.append(source_attn_mask_L)
            base_pos_B.append(base_pos)
            source_pos_B.append(source_pos)
            base_pred_B.append(base_pred)
            source_pred_B.append(source_pred)
            base_text_str.append(base_text)
            base_label_str.append(base_label)

        base_tokens_BL, base_attn_mask_BL = custom_left_padding(
            tokenizer, base_tokens_BL, max_base_len
        )
        source_tokens_BL, source_attn_mask_BL = custom_left_padding(
            tokenizer, source_tokens_BL, max_source_len
        )

        base_tokens_BL = base_tokens_BL.to(model.device)
        base_attn_mask_BL = base_attn_mask_BL.to(model.device)
        source_tokens_BL = source_tokens_BL.to(model.device)
        source_attn_mask_BL = source_attn_mask_BL.to(model.device)

        base_pos_B = torch.tensor(base_pos_B).to(model.device)
        source_pos_B = torch.tensor(source_pos_B).to(model.device)
        base_pred_B = torch.tensor(base_pred_B).to(model.device)
        source_pred_B = torch.tensor(source_pred_B).to(model.device)

        base_encoding_BL = BatchEncoding(
            {
                "input_ids": base_tokens_BL,
                "attention_mask": base_attn_mask_BL,
            }
        )
        source_encoding_BL = BatchEncoding(
            {
                "input_ids": source_tokens_BL,
                "attention_mask": source_attn_mask_BL,
            }
        )

        dataloader.append(
            (
                source_text_str,
                base_entity_labels,
                source_entity_labels,
                base_attr_types,
                source_attr_types,
                base_attr_labels,
                source_attr_labels,
                base_encoding_BL,
                source_encoding_BL,
                base_pos_B,
                source_pos_B,
                base_pred_B,
                source_pred_B,
                base_text_str,
                base_label_str,
            )
        )

    print(f"Created dataloader with {len(dataloader)} batches")
    return dataloader


def run_eval_single_cause_attribute(
    dataset,
    entity_class,
    cause_attribute: str,
    iso_attributes: list[str],
    config: RAVELEvalConfig,
    sae: SAE,
    model: PreTrainedModel,
    sae_result_path: str,
) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    cause_base_prompts, cause_source_prompts = get_prompt_pairs(
        dataset=dataset,
        base_attribute=cause_attribute,
        source_attribute=cause_attribute,
        n_interventions=config.num_pairs_per_attribute,
    )

    iso_base_prompts = []
    iso_source_prompts = []
    for iso_attr in iso_attributes:
        attr_base_prompts, attr_source_prompts = get_prompt_pairs(
            dataset=dataset,
            base_attribute=iso_attr,
            source_attribute=iso_attr,
            n_interventions=config.num_pairs_per_attribute,
        )
        iso_base_prompts.extend(attr_base_prompts)
        iso_source_prompts.extend(attr_source_prompts)

    combined = list(zip(iso_base_prompts, iso_source_prompts))
    random.seed(config.random_seed)
    random.shuffle(combined)
    iso_base_prompts, iso_source_prompts = zip(*combined)

    # Truncate to match the length of cause prompts
    cause_length = len(cause_base_prompts)
    iso_base_prompts = list(iso_base_prompts[:cause_length])
    iso_source_prompts = list(iso_source_prompts[:cause_length])

    print(
        f"Using {len(cause_base_prompts)} cause prompt pairs and {len(iso_base_prompts)} ISO prompt pairs"
    )

    train_loader, val_loader = create_dataloaders(
        cause_base_prompts,
        cause_source_prompts,
        iso_base_prompts,
        iso_source_prompts,
        model,
        config,
        train_test_split=config.train_test_split,
    )

    # dataloaders with extra attribute information for passing on to generate batched interventions
    train_loader_extended, val_loader_extended = create_dataloaders_extended(
        cause_base_prompts,
        cause_source_prompts,
        iso_base_prompts,
        iso_source_prompts,
        model,
        config,
        train_test_split=config.train_test_split,
    )

    torch.cuda.empty_cache()
    gc.collect()

    mdbm_save_dir = _mdbm_result_dir(config.mdbm_dir, sae_result_path)
    os.makedirs(mdbm_save_dir, exist_ok=True)

    downsample = config.full_dataset_downsample
    top_n_entities = config.top_n_entities
    mdbm_name = f"{entity_class}_{cause_attribute}_downsampled-{downsample}_top-{top_n_entities}.pt"
    mdbm_save_path = os.path.join(mdbm_save_dir, mdbm_name)

    trained_mdbm = mdbm.train_mdbm(
        model,
        tokenizer,
        config,
        sae,
        train_loader=train_loader,
        val_loader=val_loader,
        verbose=False,
        train_mdas=config.train_mdas,
    )

    _save_mdbm_checkpoint(trained_mdbm, mdbm_save_path)
    torch.cuda.empty_cache()
    gc.collect()

    iso_score, cause_score, isolation_instances, cause_instances = (
        intervention.generate_batched_interventions(
            model,
            trained_mdbm,
            tokenizer,
            # val_loader,  # type: ignore
            val_loader_extended,  # changed to val_loader_extended
            max_new_tokens=config.n_generated_tokens,
        )
    )

    return {
        "cause_score": cause_score,  # float
        "isolation_score": iso_score,  # float
        "disentangle_score": (cause_score + iso_score) / 2,  # float
        "isolation_instances": isolation_instances,  # list of dicts
        "cause_instances": cause_instances,  # list of dicts
    }


def run_eval_single_dataset(
    entity_class: str,
    config: RAVELEvalConfig,
    sae: SAE,
    model: PreTrainedModel,
    sae_result_path: str,
) -> tuple[dict[str, float], dict]:
    """config: eval_config.EvalConfig contains all hyperparameters to reproduce the evaluation.
    It is saved in the results_dict for reproducibility."""

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    filtered_dataset_filename = get_instance_name(
        entity_class,
        config.model_name,
        config.full_dataset_downsample,
        config.top_n_entities,
    )
    filtered_dataset_path = os.path.join(config.artifact_dir, filtered_dataset_filename)

    if not os.path.exists(filtered_dataset_path):
        orig_batch_size = config.llm_batch_size
        # Generations use much less memory than training the MDBM
        config.llm_batch_size = orig_batch_size * 8
        full_dataset = RAVELInstance.create_from_files(
            config=config,
            entity_type=entity_class,
            tokenizer=tokenizer,
            data_dir=config.artifact_dir,
            model=model,
            model_name=config.model_name,
            attribute_types=config.entity_attribute_selection[entity_class],
            downsample=config.full_dataset_downsample,
        )
        config.llm_batch_size = orig_batch_size

        # Create filtered dataset.
        full_dataset.create_and_save_filtered_dataset(
            artifact_dir=config.artifact_dir,
            top_n_entities=config.top_n_entities,
        )

    # Test loading the filtered dataset.
    dataset = RAVELFilteredDataset.load(filtered_dataset_path)
    ##########################

    attributes = config.entity_attribute_selection[entity_class]

    results_dict = {
        "cause_score": [],
        "isolation_score": [],
        "disentangle_score": [],
        "isolation_instances": [],
        "cause_instances": [],
    }
    per_class_results_dict = {}

    for cause_attribute in attributes:
        iso_attributes = [attr for attr in attributes if attr != cause_attribute]

        gc.collect()
        torch.cuda.empty_cache()

        mdbm_results = run_eval_single_cause_attribute(
            dataset,
            entity_class,
            cause_attribute,
            iso_attributes,
            config,
            sae,
            model,
            sae_result_path,
        )

        print(mdbm_results)

        results_dict["cause_score"].append(
            mdbm_results["cause_score"]
        )  # list of floats
        results_dict["isolation_score"].append(
            mdbm_results["isolation_score"]
        )  # list of floats
        results_dict["disentangle_score"].append(
            mdbm_results["disentangle_score"]
        )  # list of floats
        results_dict["isolation_instances"].extend(
            mdbm_results["isolation_instances"]
        )  # remains a list of dicts
        results_dict["cause_instances"].extend(
            mdbm_results["cause_instances"]
        )  # remains a list of dicts

        per_class_results_dict[f"{entity_class}_{cause_attribute}"] = mdbm_results

    for key in results_dict.keys():
        if key in ["cause_score", "isolation_score", "disentangle_score"]:
            results_dict[key] = sum(results_dict[key]) / len(results_dict[key])  # type: ignore

    return results_dict, per_class_results_dict  # type: ignore


def run_eval_single_sae(
    config: RAVELEvalConfig,
    sae: SAE,
    model: PreTrainedModel,
    device: str,
    artifacts_folder: str,
    sae_result_path: str,
) -> tuple[dict[str, float | dict[str, float]], dict]:
    """NOTE: This is currently setup for Transformers, not TransformerLens models."""

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    os.makedirs(artifacts_folder, exist_ok=True)
    torch.set_grad_enabled(True)

    results_dict = {}

    dataset_results = {}
    per_class_dict = {}
    for entity_class in config.entity_attribute_selection.keys():
        (
            dataset_results[f"{entity_class}_results"],
            per_class_dict[f"{entity_class}_results"],
        ) = run_eval_single_dataset(
            entity_class,
            config,
            sae,
            model,
            sae_result_path,
        )

    results_dict = general_utils.average_results_dictionaries(
        dataset_results, list(config.entity_attribute_selection.keys())
    )

    for entity_class, dataset_result in dataset_results.items():
        results_dict[f"{entity_class}"] = dataset_result

    return results_dict, per_class_dict  # type: ignore


def run_eval(
    config: RAVELEvalConfig,
    selected_saes: list[tuple[str, SAE]] | list[tuple[str, str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    artifacts_path: str = "data/artifacts",
):
    """
    selected_saes is a list of either tuples of (sae_lens release, sae_lens id) or (sae_name, SAE object)

    Return dict is a dict of SAE name: evaluation results for that SAE."""
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    gc.collect()
    torch.cuda.empty_cache()

    artifacts_folder = None
    os.makedirs(output_path, exist_ok=True)

    results_dict = {}

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    config.model_name = LLM_NAME_MAP[config.model_name]

    if "gemma" in config.model_name:
        model_kwargs = {"attn_implementation": "eager"}
    else:
        model_kwargs = {}

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=device,
        torch_dtype=llm_dtype,
        **model_kwargs,
    )

    for sae_release, sae_object_or_id in tqdm(
        selected_saes, desc="Running SAE evaluation on all selected SAEs"
    ):
        sae_id, sae, sparsity = general_utils.load_and_format_sae(
            sae_release, sae_object_or_id, device
        )  # type: ignore
        sae = sae.to(device=device, dtype=llm_dtype)

        if config.train_mdas:
            sae_release = "mdas"
            sae_id = "mdas"
            assert len(selected_saes) == 1

        sae_result_path = general_utils.get_results_filepath(
            output_path, sae_release, sae_id
        )

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {sae_release}_{sae_id} as results already exist")
            continue

        artifacts_folder = os.path.join(
            artifacts_path,
            EVAL_TYPE_ID_RAVEL,
            config.model_name,
            sae.cfg.hook_name,
        )

        eval_results, per_class_dict = run_eval_single_sae(
            config, sae, model, device, artifacts_folder, sae_result_path
        )
        eval_output = RAVELEvalOutput(
            eval_config=config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
            eval_result_metrics=RAVELMetricCategories(
                ravel=RAVELMetricResults(
                    disentanglement_score=eval_results["disentangle_score"],  # type: ignore
                    cause_score=eval_results["cause_score"],  # type: ignore
                    isolation_score=eval_results["isolation_score"],  # type: ignore
                )
            ),
            # eval_result_details needs to be a list for compatibility with other evals,
            # but for RAVEL we are just putting all the details in the unstructured field
            eval_result_details=[],
            eval_result_unstructured={
                "per_class": per_class_dict,
                "raw_results": eval_results,
            },
            sae_bench_commit_hash=sae_bench_commit_hash,
            sae_lens_id=sae_id,
            sae_lens_release_id=sae_release,
            sae_lens_version=sae_lens_version,
            sae_cfg_dict=asdict(sae.cfg),
        )

        results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)

        eval_output.to_json_file(sae_result_path, indent=4)

        gc.collect()
        torch.cuda.empty_cache()

    return results_dict


def create_config_and_selected_saes(
    args,
) -> tuple[RAVELEvalConfig, list[tuple[str, str]]]:
    config = RAVELEvalConfig(
        model_name=args.model_name,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        # ctx len here is usually around 32, so we can use a larger batch size
        # However, we do have backward passes for training the MDBM
        # The divide by 4 shouldn't be necessary, but there's some memory fragmentation issue
        # that causes intermittent OOM errors.
        config.llm_batch_size = (
            activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name] // 4
        )

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    if args.random_seed is not None:
        config.random_seed = args.random_seed

    if args.train_mdas:
        config.train_mdas = args.train_mdas
        config.num_epochs = 10

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(description="Run RAVEL evaluation")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--sae_regex_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE selection",
    )
    parser.add_argument(
        "--sae_block_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE block selection",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/instance_eval_results/ravel",
        help="Output folder",
    )
    parser.add_argument(
        "--force_rerun", action="store_true", help="Force rerun of experiments"
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Batch size for LLM. If None, will be populated using LLM_NAME_TO_BATCH_SIZE",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, will be populated using LLM_NAME_TO_DTYPE",
    )
    parser.add_argument(
        "--artifacts_path",
        type=str,
        default="data/artifacts",
        help="Path to save artifacts",
    )
    parser.add_argument(
        "--train_mdas",
        action="store_true",
        help="Train MDAS instead of SAEs",
    )

    return parser


if __name__ == "__main__":
    """
    python -m sae_bench.evals.ravel.main \
    --sae_regex_pattern "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109" \
    --sae_block_pattern "blocks.12.hook_resid_post__trainer_2" \
    --model_name gemma-2-2b
    """
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)

    print(selected_saes)

    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes,
        device,
        args.output_folder,
        args.force_rerun,
        artifacts_path=args.artifacts_path,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")


# Use this code snippet to use custom SAE objects
# if __name__ == "__main__":
#     import sae_bench.custom_saes.identity_sae as identity_sae
#     import sae_bench.custom_saes.jumprelu_sae as jumprelu_sae

#     """
#     python evals/ravel/main.py
#     """
#     device = general_utils.setup_environment()

#     start_time = time.time()

#     random_seed = 42
#     output_folder = "eval_results/ravel"

#     model_name = "gemma-2-2b"
#     hook_layer = 20

#     repo_id = "google/gemma-scope-2b-pt-res"
#     filename = f"layer_{hook_layer}/width_16k/average_l0_71/params.npz"
#     sae = jumprelu_sae.load_jumprelu_sae(repo_id, filename, hook_layer)
#     selected_saes = [(f"{repo_id}_{filename}_gemmascope_sae", sae)]

#     config = RAVELEvalConfig(
#         random_seed=random_seed,
#         model_name=model_name,
#     )

#     config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
#     config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

#     # create output folder
#     os.makedirs(output_folder, exist_ok=True)

#     # run the evaluation on all selected SAEs
#     results_dict = run_eval(
#         config,
#         selected_saes,
#         device,
#         output_folder,
#         force_rerun=True,
#     )

#     end_time = time.time()

#     print(f"Finished evaluation in {end_time - start_time} seconds")
