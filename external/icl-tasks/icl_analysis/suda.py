from types import SimpleNamespace as Namespace
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm
import gc

import nltk
import torch
import math
import numpy as np
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset


import scipy

torch.set_grad_enabled(False)

from icl_analysis.induction_dataset import InductionDataset

from sklearn.metrics import f1_score 


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import ConnectionPatch

torch.set_grad_enabled(False)

def get_dataset_activation_caches(model, prompts):
    caches = []

    for idx, prompt in enumerate(prompts):
        _, cache = model.run_with_cache(prompt, remove_batch_dim=True, pos_slice=-1)
        cache.to("cpu")
        caches.append(cache)

    return caches

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('split', type=int)
    parser.add_argument('vh_path')
    parser.add_argument('output_path')
    args = parser.parse_args()
    s = args.split

    steps = [2 ** x for x in range(10)] + list(range(1000, 144000, 1000))
    path = Path(args.vh_path)

    steps.remove(50000)

    steps = steps[s*40:(s+1)*40]
    print('Running steps:', *steps)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")

    config = Namespace(seed=0)
    ds = InductionDataset(config, tokenizer)
    ds.parse_brown()  # Parse the brown dataset for random tokens

    data = {}

    data['lsc'] = ds.generate_s1(n_sample=1000)
    data['lscg'] = ds.generate_s2(n_sample=1000)
    data['wi'] = ds.generate_r1(n_sample=1000)
    data['wc'] = ds.generate_wc2(n_sample=1000)
    data['translate'] = ds.generate_translate('en', 'de', n_examples=1000)
    data['counterfactual'] = ds.generate_cf(n_sample=1000)

    random.shuffle(steps)

    for step in tqdm(steps):
        avg_logits, avg_probs = {}, {}

        inference_dtype = torch.bfloat16
        model = HookedTransformer.from_pretrained(
            f"EleutherAI/pythia-12b",
            revision=f'step{step}',
            torch_dtype=inference_dtype, fold_ln=False, center_writing_weights=False, center_unembed=False)
        
        Vh = torch.load(path / f'pythia_12b_step{step}.pt')

        W_U = model.W_U.to(0)
        WU_T = W_U.transpose(0, 1)
        C_global = torch.matmul(WU_T, Vh.T)

        for task, (prompts, answers) in data.items():

            caches = get_dataset_activation_caches(model, prompts)

            logits_lst, probs_lst = [], []

            for cache_idx, cache in enumerate(caches):

                answer_token_ids = model.tokenizer.encode(answers[cache_idx], add_special_tokens=False)
                answer_token_id = answer_token_ids[0]

                activation = cache["resid_post", -1][-1, :].to(0)

                A = torch.matmul(activation, Vh.T).float()  # [num_sv]
                logits = (C_global * A.unsqueeze(0))
                answer_logits = logits[answer_token_id, :]
                probs = logits.softmax(dim=0)
                answer_probs = probs[answer_token_id, :]  # [num_sv]

                logits_lst.append(answer_logits.cpu())
                probs_lst.append(answer_probs.cpu())

            avg_logits[task] = torch.stack(logits_lst).mean(0)
            avg_probs[task] = torch.stack(probs_lst).mean(0)

        try:
            del model
        except NameError:
            pass

        gc.collect()
        torch.cuda.empty_cache()

        torch.save([avg_logits, avg_probs], f'{args.output_path}/pythia_12b_{step}.pt')