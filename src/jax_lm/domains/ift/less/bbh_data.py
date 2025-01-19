import glob
import tqdm
import os
import json
import random
from pathlib import Path

# originally 40
def get_bbh_test(data_dir, max_num_examples_per_task=None):
    assert max_num_examples_per_task is None
    rng = random.Random(0)
    data_dir = Path(data_dir) / 'eval/bbh'
    all_tasks = {}
    tfp = str(data_dir / "test" / "*.json")
    task_files = list(glob.glob(tfp))
    assert len(task_files) > 0, f"No task files found matching pattern {tfp}."
    for task_file in tqdm.tqdm(task_files, desc="Loading tasks"):
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]
            if max_num_examples_per_task:
                all_tasks[task_name] = rng.sample(
                    all_tasks[task_name], max_num_examples_per_task)

    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join(data_dir, "cot-prompts", "*.txt"))
    for cot_prompt_file in tqdm.tqdm(cot_prompt_files, desc="Loading prompts"):
        with open(cot_prompt_file, "r") as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            task_prompt = "".join(f.readlines()[2:])
            all_prompts[task_name] = task_prompt

    all_examples = []
    def sample_to_tuple(x, task_name):
        icl = all_prompts[task_name]
        # icl_samples = '\n\n'.join()
        icl_samples = list(icl.split('\n\n')[-3:])
        icl_header = icl.split('\n\n')[0].strip()
        question = x['input']
        answer = x['target']
        tup = (icl_header, question, answer, icl_samples, task_name)
        # print('>> TUP:', tup)
        # import pdb; pdb.set_trace()
        return tup

    for task_name, examples in all_tasks.items():
        for x in examples:
            all_examples.append(sample_to_tuple(x, task_name))

    return all_examples
