import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import sae_bench.evals.ravel.mdbm as mdbm
import sae_bench.sae_bench_utils.activation_collection as activation_collection
from sae_bench.evals.ravel.instance import (
    Prompt,
    RAVELFilteredDataset,
    evaluate_completion,
)


def get_different_attribute_prompt(
    base_prompt: Prompt, source_prompts: list[Prompt]
) -> Prompt:
    """
    Select a random prompt from source_prompts that has a different attribute_label
    than the base_prompt.
    """
    different_prompts = [
        p for p in source_prompts if p.attribute_label != base_prompt.attribute_label
    ]
    if not different_prompts:
        raise ValueError(
            f"No prompts with different attribute label found for {base_prompt.attribute_label}"
        )
    return random.choice(different_prompts)


def sample_prompts_by_attribute(
    dataset: RAVELFilteredDataset, attribute: str, n_samples: int
):
    all_prompts = dataset.get_prompts_by_attribute(attribute)
    if len(all_prompts) < n_samples:
        print(
            f"Warning: Not enough prompts for attribute {attribute} for intervention. Returning {len(all_prompts)} instead of {n_samples} prompts."
        )
        return all_prompts, all_prompts

    selected_prompts = random.sample(all_prompts, n_samples)
    return all_prompts, selected_prompts


def get_prompt_pairs(
    dataset: RAVELFilteredDataset,
    base_attribute: str,
    source_attribute: str,
    n_interventions: int,
):
    """
    Selects pairs of base_prompts and source_prompts for the cause and isolation evaluations.
    Base_prompts always contain attribute A templates.
    The cause evaluation requires source_prompts from attribute A templates, attribute values in base and source should differ.
    The isolation evaluation requires source_prompts from attribute B templates.
    """
    all_base_prompts, base_prompts = sample_prompts_by_attribute(
        dataset, base_attribute, n_interventions
    )

    if base_attribute != source_attribute:
        _, source_prompts = sample_prompts_by_attribute(
            dataset, source_attribute, n_interventions
        )
    else:
        all_source_prompts = all_base_prompts
        source_prompts = []
        for p in base_prompts:
            source_prompts.append(get_different_attribute_prompt(p, all_source_prompts))

    min_length = min(len(base_prompts), len(source_prompts))
    return base_prompts[:min_length], source_prompts[:min_length]


@torch.no_grad()
def generate_batched_interventions(
    model: PreTrainedModel,
    mdbm: mdbm.MDBM,
    tokenizer: PreTrainedTokenizerBase,
    val_loader: torch.utils.data.DataLoader,
    max_new_tokens: int = 8,
) -> tuple[float, float, list, list]:
    iso_scores = []
    cause_scores = []
    # to store information about the individual instances used for isolation and cause score calculations
    isolation_instances=[]
    cause_instances=[]

    for batch in tqdm(val_loader, desc="Generating with interventions"):
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
        ) = batch #each element of the tuple is a list representing the attributes for the datapoints in the batch

        # Get source representation
        source_rep_BD = activation_collection.get_layer_activations(
            model, mdbm.layer_intervened, source_encoding_BL, source_pos_B
        )

        intervention_hook = mdbm.create_intervention_hook(
            source_rep_BD,
            base_pos_B,
            training_mode=False,
        )

        handle = activation_collection.get_module(
            model, mdbm.layer_intervened
        ).register_forward_hook(intervention_hook)

        # Generate using huggingface model
        output_ids = model.generate(
            input_ids=base_encoding_BL["input_ids"].to(model.device),
            attention_mask=base_encoding_BL.get("attention_mask", None).to(
                model.device
            ),
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding for reproducibility
        )  # type: ignore

        handle.remove()
        generated_ids = output_ids[:, -max_new_tokens:]
        generated_strings = tokenizer.batch_decode(generated_ids)

        for source_text, base_entity_label, source_entity_label, base_attr_type, source_attr_type, base_attr_label, source_attr_label, base_pos, source_pos, base_input_ids, base_attention_mask, source_input_ids, source_attention_mask, base_text, base_label, generated_string, base_pred, source_pred in zip(
            source_text_str, base_entity_labels, source_entity_labels, base_attr_types, source_attr_types, base_attr_labels, source_attr_labels, base_pos_B.cpu().tolist(), source_pos_B.cpu().tolist(), base_encoding_BL["input_ids"].cpu().tolist(), base_encoding_BL["attention_mask"].cpu().tolist(), source_encoding_BL["input_ids"].cpu().tolist(), source_encoding_BL["attention_mask"].cpu().tolist(), base_text_str, base_label_str, generated_strings, base_pred_B, source_pred_B
        ):
            iso_instance={}
            cause_instance={}

            if base_pred == source_pred:
                iso_score = evaluate_completion(base_text, base_label, generated_string)
                iso_scores.append(iso_score)
                
                #dict
                iso_instance={
                    "base_prompt": base_text, 
                    "base_input_ids": base_input_ids,
                    "base_attention_mask": base_attention_mask,
                    "base_entity_label": base_entity_label,
                    "base_attribute_type": base_attr_type,
                    "base_attribute_label": base_attr_label,
                    "base_entity_final_token_pos": base_pos,
                    "base_prediction": base_pred,
                    "source_prompt": source_text,
                    "source_input_ids": source_input_ids,
                    "source_attention_mask": source_attention_mask,
                    "source_entity_label": source_entity_label,
                    "source_attribute_type": source_attr_type,
                    "source_attribute_label": source_attr_label,
                    "source_entity_final_token_pos": source_pos,
                    "source_prediction": source_pred,
                    "corrupted_prediction": generated_string,
                    "cause_score": -1, #since we are checking for isolation here
                    "isolation_score": int(iso_score), 
                }

                isolation_instances.append(iso_instance)

            else:
                cause_score = evaluate_completion(base_text, base_label, generated_string)
                cause_scores.append(cause_score)

                #dict
                cause_instance={
                    "base_prompt": base_text, 
                    "base_input_ids": base_input_ids,
                    "base_attention_mask": base_attention_mask,
                    "base_entity_label": base_entity_label,
                    "base_attribute_type": base_attr_type,
                    "base_attribute_label": base_attr_label,
                    "base_entity_final_token_pos": base_pos,
                    "base_prediction": base_pred,
                    "source_prompt": source_text,
                    "source_input_ids": source_input_ids,
                    "source_attention_mask": source_attention_mask,
                    "source_entity_label": source_entity_label,
                    "source_attribute_type": source_attr_type,
                    "source_attribute_label": source_attr_label,
                    "source_entity_final_token_pos": source_pos,
                    "source_prediction": source_pred,
                    "corrupted_prediction": generated_string,
                    "cause_score": int(cause_score), 
                    "isolation_score": -1, #since we are checking for cause here
                }
                cause_instances.append(cause_instance)

    mean_iso_score = float(np.mean(iso_scores)) #average isolation score for a single cause attribute of an entity dataset
    mean_cause_score = float(np.mean(cause_scores)) #average cause score for a single cause attribute of an entity dataset
    return mean_iso_score, mean_cause_score, isolation_instances, cause_instances