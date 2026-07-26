import argparse
from pathlib import Path
import gc

from tqdm.auto import tqdm

import torch
torch.set_grad_enabled(False)

from transformer_lens import HookedTransformer

def clean():
    try:
        del model
    except NameError:
        pass

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('split', type=int)
    parser.add_argument('vh_path')
    args = parser.parse_args()
    s = args.split

    steps = [2 ** x for x in range(10)] + list(range(1000, 144000, 1000))

    steps = steps[s*40:(s+1)*40]
    print('Running steps:', *steps)

    path = Path(args.vh_path)

    for step in tqdm(steps):
        inference_dtype = torch.bfloat16
        model = HookedTransformer.from_pretrained(
            f"EleutherAI/pythia-12b",
            revision=f'step{step}',
            torch_dtype=inference_dtype, fold_ln=False, center_writing_weights=False, center_unembed=False)

        _, _, Vh = torch.linalg.svd(model.W_U.type(torch.float).T.cpu())

        Vh = Vh.type(torch.bfloat16).cuda()
        vh_file_path = path / f'pythia_12b_step{step}.pt'
        torch.save(Vh, vh_file_path)
        print(step, 'done!')
        clean()