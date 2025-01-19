import json
import numpy as np
import os
from typing import List, Tuple
from collections import defaultdict
# from jaxlm.less.tydiqa_data import get_test_data

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"

def tokenize(tokenizer: PreTrainedTokenizerBase,
             query: str,
             completion: str,
             max_length: int,
             print_ex: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
        query (str): The question part of the chat conversation.
        completion (str): The answer part of the chat conversation.
        max_length (int): The maximum length of the input tensors.
        print_ex (bool, optional): Whether to print the example. Defaults to False.

    Returns:
        tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
    """
    full_prompt = query + completion

    if print_ex:
        print("******** Example starts ********")
        print(full_prompt)
        print("******** Example ends ********")

    prompt_input_ids = torch.tensor(
        tokenizer.encode(query, max_length=max_length))
    full_input_ids = torch.tensor(
        tokenizer.encode(full_prompt, max_length=max_length))
    # import pdb; pdb.set_trace()
    labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
    labels[:len(prompt_input_ids)] = -100
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask


def get_bbh_dataset(data_dir: str,
                    tokenizer: PreTrainedTokenizerBase,
                    max_length: int,
                    use_chat_format,
                    chat_format,
                    partition,
                    *,
                    max_response: int,
                    **kwargs):
    """
    Get the bbh dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Ex1>
    <Ex2>
    <Question of Ex3>
    <|assistant|>
    A:

    Completion:
    <Answer of Ex3>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the input. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        n_shot (int, optional): The number of shots for few-shot learning. Defaults to 3 for bbh.

    Returns:
        Dataset: The BBH dataset containing input_ids, attention_mask, and labels.
    """
    max_length = set_max_length(max_length, partition,
                                max_response=max_response)
    # TEST: has 40 max samples per category
    file = f"{data_dir}/eval/bbh/bbh-three-shot.json"
    bbh_few_shot_examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": [], 'category': []}

    # there are multiple tasks in the bbh dataset
    # each task has 3 examples
    def form_icl(exs):
        string = ""
        for ex in exs:
            question, answer = ex.split("\nA:")
            string += question + "\nA:" + answer
            answer = ' ' + answer
            assert answer[0] == ' '
            string += "\n\n"
        return string

    n, skipped, maxlength = 0, 0, -1

    def make_val():
        all_samples = []
        for task in bbh_few_shot_examples:
            few_shot_exs = bbh_few_shot_examples[task]

            stuff = few_shot_exs.split("\n\n")
            exes = stuff[-3:]
            task_prompt = "\n\n".join(stuff[:-3])

            for i in range(len(exes)):
                # these are all strings
                target_ex = exes[i]
                other_exes = exes[:i] + exes[i+1:]
                Q, A = target_ex.split("\nA:")
                assert A[0] == ' '
                # all_samples.append((task_prompt, question, answer, other_exes))
                all_samples.append((task_prompt, Q, A, other_exes, task))

        return all_samples

    def make_test():
        from .bbh_data import get_bbh_test
        return get_bbh_test(data_dir)

    # val_samples, test_samples = make_val(), make_test()
    # import pdb; pdb.set_trace()

    samples = make_val() if partition == 'val' else make_test()
    assert partition in ['test', 'val']

    # ONE SHOT NOW
    rng = np.random.default_rng(0)
    for task_prompt, question, answer, icl_fss, task in samples:
        assert isinstance(icl_fss, list)

        def build_prompt_for_fss(fss, question, answer, task, task_prompt):
            icl = form_icl(fss)
        # for task_prompt, target_ex, other_exes in make_val():
            # icl = '\n\n'.join(icl.split('\n\n')[-1:])
            baselining = False
            icl = icl.strip()
            icl = icl + '\n\n'
            if use_chat_format and baselining is False:
                if chat_format == "tulu":
                    question = "<|user|>\n" + task_prompt.strip() + "\n\n" + icl + \
                        f"{question}" + "\n<|assistant|>\nA:"
                else:
                    raise NotImplementedError
                    question = f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
            else:
                question = task_prompt.strip() + "\n\n" + icl + \
                    f"{question}" + "\nA:"

            # assert answer[0] == ' ', answer
            if not answer[0] == ' ':
                answer = ' ' + answer

            if partition == 'test':
                answer = answer + '|| ' + task

            # if partition != 'test':

            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, question, answer, max_length + 1, print_ex=False)
            import torch as ch
            full_input_ids = ch.concat([full_input_ids, ch.tensor([tokenizer.eos_token_id])])
            labels = ch.concat([labels, ch.tensor([tokenizer.eos_token_id])])
            attention_mask.append(1)
            return full_input_ids, labels, attention_mask

        # longest first
        sorted_icl = sorted(icl_fss, key=lambda x: -len(x))
        num_dropped = 0
        while True:
            taken_icl = sorted_icl[num_dropped:]
            shuffled_icl = rng.permutation(len(taken_icl))
            shuffled_icl = [taken_icl[int(i)] for i in shuffled_icl]
            out = build_prompt_for_fss(shuffled_icl, question, answer, task,
                                       task_prompt)
            full_input_ids, labels, attention_mask = out
            if partition == 'test': # then need to have space to generate cot
                if len(full_input_ids) <= max_length - 1024:
                    break
            else:
                if len(full_input_ids) <= max_length:
                    break

            num_dropped += 1

        if num_dropped == len(sorted_icl):
            print('DROPPED FULL EXAMPLE')

        maxlength = max(maxlength, len(full_input_ids))
        n += 1
        if len(full_input_ids) > max_length + 1:
            skipped += 1
            continue
        else:
            if any(labels != -100):
                dataset["input_ids"].append(full_input_ids[:-1])
                dataset["labels"].append(labels[1:])
                dataset["attention_mask"].append(attention_mask[1:])
                dataset['category'].append(task)
            else:
                skipped += 1
                maxlength = max(maxlength, len(full_input_ids))

    print('BBH: Skipped {} examples of {} total with max length {}'.format(skipped, n, maxlength))

    dataset = Dataset.from_dict(dataset)
    return dataset

def set_max_length(max_length, partition, *, max_response=1024):
    assert max_length == 3072
    assert max_response == 1024
    if partition == 'test':
        max_length = max_length - max_response
    else:
        max_length = max_length

    return max_length

def get_mmlu_dataset(data_dir: str,
                     tokenizer: PreTrainedTokenizerBase,
                     max_length: int,
                     use_chat_format,
                     chat_format,
                     partition,
                     *,
                     max_response,
                     **kwargs):
    """
    Get the MMLU dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Question>
    <|assistant|>
    The answer is:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """
    max_length = set_max_length(max_length, partition, max_response=max_response)
    # max_length = 2048
    mmlu_data_dir = os.path.join(data_dir, "eval", "mmlu")
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(mmlu_data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry

        return s.strip()

    rng = np.random.default_rng(0)

    def gen_prompt(train_df, subject, i=0, prompt_df=None, exclude=[], tokenizer=None):
        assert tokenizer is not None
        start_prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        few_shot_prompts = []
        if prompt_df is not None:
            for j in range(5):
                if j not in exclude:
                    pp = format_example(prompt_df, j, include_answer=True)
                    few_shot_prompts.append(pp)

        final_example = format_example(train_df, i, include_answer=False)

        def build_prompt(fss, fs):
            prompt = start_prompt
            for i, f in enumerate(list(fss)):
                prompt += f + "\n\n"

            prompt += fs
            return prompt

        def tokens_for_prompt(fss):
            prompt = build_prompt(fss, final_example)
            encoded = tokenizer.encode(prompt)
            if 'input_ids' in encoded and isinstance(encoded, dict):
                encoded = encoded['input_ids']

            if len(encoded) == 1:
                encoded = encoded[0]

            return len(encoded)

        # largest to smallest
        fss_in_order = sorted(few_shot_prompts, key=lambda x: -len(x))
        num_to_remove = 0
        while True:
            num_tokens = tokens_for_prompt(fss_in_order[num_to_remove:])
            if num_tokens <= max_length - 128:
                break

            num_to_remove += 1
            # print('> num_to_remove:', num_to_remove)
            if num_to_remove == len(fss_in_order):
                break

        fss_keep = fss_in_order[num_to_remove:]
        randomized_order = rng.permutation(len(fss_keep))
        final_prompt = build_prompt([fss_keep[i] for i in randomized_order],
                                    final_example)
        return final_prompt

    def format_example(df, idx, include_answer=True):
        choices = ["A", "B", "C", "D"]
        prompt = df.iloc[idx, 0].strip()
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            answer = ' ' + df.iloc[idx, df.shape[1] - 2 + 1]
            prompt += answer

        return prompt

    k = 5 if partition == 'val' else 50
    dataset = {"input_ids": [], "attention_mask": [], "labels": [], 'category': []}
    skipped = 0
    n = 0
    maxlength = -1

    part_pf = 'dev' if partition == 'val' else 'test'


    for subject in subjects:
        prompt_df = pd.read_csv(os.path.join(mmlu_data_dir, 'dev', subject + f"_{'dev'}.csv"), header=None)[: 5]
        dev_df = pd.read_csv(os.path.join(mmlu_data_dir, part_pf, subject + f"_{part_pf}.csv"), header=None)
        this_k = dev_df.shape[0]
        for i in range(this_k):
            if partition == 'val':
                prompt = gen_prompt(dev_df, subject, i, dev_df, [i], tokenizer=tokenizer)
            else:
                prompt = gen_prompt(dev_df, subject, i, prompt_df, tokenizer=tokenizer)

            answer = " " + dev_df.iloc[i, dev_df.shape[1] - 2 + 1]
            if partition == 'test':
                answer = answer + '||' + ' ' + subject

            assert use_chat_format and chat_format in ['tulu']
            baselining = False
            if use_chat_format and baselining is False:
                if chat_format == "tulu":
                    prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
                else:
                    prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
            else:
                prompt = prompt

            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, prompt, answer, max_length + 1, print_ex=False)

            if any(labels != -100):
                dataset["input_ids"].append(full_input_ids[:-1])
                dataset["labels"].append(labels[1:])
                dataset["attention_mask"].append(attention_mask[1:])
                dataset['category'].append(subject)
            else:
                full_input_ids_, labels_, attention_mask = tokenize(tokenizer,
                                                                  prompt,
                                                                  answer,
                                                                  9999999,
                                                                  print_ex=False)
                skipped += 1
                maxlength = max(maxlength, len(full_input_ids_))

            n += 1

    print("MMLU: Skipped {} examples of {} total with max length {} in part {}".format(skipped, n, maxlength, partition))
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_dataset(task, **kwargs):
    """
    Get the dataset for the given task.

    Args:
        task_name (str): The name of the task.

    Raises:
        ValueError: If the task name is not valid.

    Returns:
        Dataset: The dataset.
    """
    if task == "bbh":
        return get_bbh_dataset(**kwargs)
    elif task == "tydiqa":
        return get_tydiqa_dataset(**kwargs)
    elif task == "mmlu":
        return get_mmlu_dataset(**kwargs)
    else:
        raise ValueError(f"Invalid task name {task}")

def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest")
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader
