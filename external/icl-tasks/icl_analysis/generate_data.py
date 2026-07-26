import random
import json
import itertools
from functools import wraps
from nltk.corpus import brown

import torch

def reset_seed(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        random.seed(self.configs.seed)
        torch.manual_seed(self.configs.seed)
        return func(self, *args, **kwargs)
    return wrapper

class InductionDataset:
    def __init__(self, configs, tokenizer):
        self.configs = configs
        self.tokenizer = tokenizer
        self.add_space = self.auto_add_space()

    def auto_add_space(self):

        if 'llama-3.1' in self.tokenizer.name_or_path.lower():
            return True
        elif 'llama-3.2' in self.tokenizer.name_or_path.lower():
            return True
        elif 'gpt2' in self.tokenizer.name_or_path.lower():
            return True
        elif 'llama-2' in self.tokenizer.name_or_path.lower():
            return False
        elif 'pythia' in self.tokenizer.name_or_path.lower():
            return True
        else:
            raise ValueError

    @reset_seed
    def parse_brown(self, n=1000):
        def gen_1token_words():
            for word in set(brown.words()):
                if word == "''":
                    continue # this '' token will cause trouble
                t = self.tokenizer.tokenize(' '*self.add_space + word)
                if len(t) == 1:
                    yield word, self.tokenizer.convert_tokens_to_ids(t[0])
        words_gen = gen_1token_words()

        self.tokens = []
        self.token_ids = []
        for _ in range(n):
            token, token_id = next(words_gen)
            self.tokens.append(token)
            self.token_ids.append(token_id)

    @reset_seed
    def generate_lsc(self, n_sample=1000, n_source=5, n_rand=10, n_random_gap=0):
        
        # S: source
        # T: target
        # R: random
        # G: random gap
        # S1 S2 S3 G1 G2 T R... S1 S2 S3 G3 G4 T?

        result_prompts, result_labels = [], []

        for _ in range(n_sample):
            target = [random.choice(self.tokens)]
            source = random.sample(self.tokens, n_source)
            rand = random.sample(self.tokens, n_rand)

            g1 = random.sample(self.tokens, n_random_gap)
            g2 = random.sample(self.tokens, n_random_gap)

            prompt = ' '*self.add_space + ' '.join(source + g1 + target + rand + source + g2)

            result_prompts.append(prompt)
            result_labels.append(' '*self.add_space + target[0])

        return result_prompts, result_labels

    @reset_seed
    def generate_lscg(self, n_sample=1000, n_source=5, n_rand=10, n_random_gap=2):
        
        # S: source
        # T: target
        # R: random
        # G: random gap
        # S1 S2 S3 G1 G2 SS T R... S1 S2 S3 G3 G4 SS T?

        result_prompts, result_labels = [], []

        tokens = self.tokens

        for _ in range(n_sample):
            target = [random.choice(tokens)]
            source = random.sample(tokens, n_source)
            rand = random.sample(tokens, n_rand)

            g1 = random.sample(tokens, n_random_gap)
            g2 = random.sample(tokens, n_random_gap)
            ss = [random.choice(tokens)]

            prompt = ' '*self.add_space + ' '.join(source + g1 + ss + target + rand + source + g2 + ss)

            result_prompts.append(prompt)
            result_labels.append(' '*self.add_space + target[0])

        return result_prompts, result_labels

    @reset_seed
    def generate_lsc_token(self, token_range, n_sample=1000, n_source=5, n_rand=10, n_random_gap=0):
        result_prompts, result_labels = [], []
        tokens = list(range(token_range[0], token_range[1]))

        for _ in range(n_sample):
            target = [random.choice(tokens)]
            source = random.sample(tokens, n_source)
            rand = random.sample(tokens, n_rand)

            g1 = random.sample(tokens, n_random_gap)
            g2 = random.sample(tokens, n_random_gap)

            prompt_ids = source + g1 + target + rand + source + g2

            result_prompts.append(prompt_ids)
            result_labels.append(target[0])

        return result_prompts, result_labels

    @reset_seed
    def generate_wc(self,
                    n_sample=1000,
                    n_demo_per_group=3, n_features_per_group=2, n_groups=2,
                    n_rand=50, n_dstr=7,
                    input_start='', label_start='', sep='\n'):

        result_prompts, result_labels = [], []

        for _ in range(n_sample):
            dstr_tokens = random.sample(self.tokens, n_rand)
            groups = {}

            labels = random.sample(self.tokens, n_groups)
            for label in labels:
                features = random.sample(self.tokens, n_features_per_group)
                groups[label] = features

            prompts = []

            for label, features in groups.items():
                for _ in range(n_demo_per_group):
                    dstrs = random.choices(dstr_tokens, k=n_dstr)
                    tokens = dstrs + features
                    random.shuffle(tokens)
                    demo = ' '.join([input_start] + tokens) + sep + label_start + ' ' + label + sep
                    prompts.append(demo)
            random.shuffle(prompts)
            dstrs = random.choices(dstr_tokens, k=n_dstr)
            tokens = dstrs + features
            question = ' '.join([input_start] + tokens) + sep + label_start

            result_prompts.append(''.join(prompts + [question]))
            result_labels.append(' '*self.add_space + label)

        return result_prompts, result_labels

    @reset_seed
    def generate_wi(self, n_sample=1000, seq_len=3, target_i=2,
                    n_examples_per_question=5,
                    sep=';', arrow='->',
        ):
        result_prompts, result_labels = [], []

        for _ in range(n_sample):

            prompt_lst = []
            for _ in range(n_examples_per_question):
                tokens = random.sample(self.tokens, seq_len)
                prompt_lst.append(' '.join(tokens) + f' {arrow} ' + tokens[target_i] + sep)

            question_tokens = random.sample(self.tokens, seq_len)
            prompt = ' '*self.add_space + ' '.join(prompt_lst) + ' ' + ' '.join(question_tokens) + f' {arrow}'

            result_prompts.append(prompt)
            result_labels.append(' '*self.add_space + question_tokens[target_i])
        
        return result_prompts, result_labels

    @reset_seed
    def generate_translate(self, lang_1, lang_2, n_examples=1000, n_demos=5):

        from .words import translations
        result_prompts, result_labels = [], []

        w1 = translations[lang_1]
        w2 = translations[lang_2]

        pairs = []
        for ww1, ww2 in zip(w1, w2):
            pairs.append((ww1, ww2))

        for _ in range(n_examples):

            query = random.choice(pairs)
            demos = random.sample([x for x in pairs if x != query], 5)

            prompt = ' '*self.add_space
            for l1_demo, w2_demo in demos:
                prompt += f'{l1_demo} -> {w2_demo}; '

            prompt += f'{query[0]} ->'
            result_prompts.append(prompt)
            result_labels.append(' '*self.add_space + query[1])

        return result_prompts, result_labels
    
    @reset_seed
    def generate_cf(self, n_sample=1000):
        '''Counterfactual'''

        with open('data/country_capital.json') as open_file:
            json_data = json.load(open_file)

        data = []

        result_prompts, result_labels = [], []

        for country, capital in json_data.items():
            capital_token = self.tokenizer.tokenize(' '*self.add_space + capital)
            if len(capital_token) != 1:
                continue
            country_token = self.tokenizer.tokenize(' '*self.add_space + country)
            if len(country_token) != 1:
                continue
            data.append((country, capital))

        combs = list(itertools.permutations(data, 2))
        if n_sample < len(combs):
            combs = random.sample(combs, n_sample)

        for (cc1, cc2) in combs:
            country_1, capital_1 = cc1
            country_2, capital_2 = cc2

            prompt = f"If we swtich the capital of {country_1} and {country_2}, then {country_1}'s capital is {capital_2} and {country_2}'s capital is"
            label = capital_1
            
            result_prompts.append(prompt)
            result_labels.append(' '*self.add_space + label)

        return result_prompts, result_labels
