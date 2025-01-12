import os
import json
import random

def get_test_data(data_dir, max_num_examples_per_lang=None): # , max_num_examples_per_lang=200):
    assert max_num_examples_per_lang is None
    test_data = []
    with open(os.path.join(data_dir, "eval/tydiqa/test/tydiqa-goldp-v1.1-dev.json")) as fin:
        dev_data = json.load(fin)
        for article in dev_data["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    example = {
                        "id": qa["id"],
                        "lang": qa["id"].split("-")[0],
                        "context": paragraph["context"],
                        "question": qa["question"],
                        "answers": qa["answers"]
                    }
                    test_data.append(example)

    data_languages = set([example["lang"] for example in test_data])
    sampled_examples = []
    for lang in data_languages:
        examples_for_lang = [
            example for example in test_data if example["lang"] == lang]
        if max_num_examples_per_lang is not None and len(examples_for_lang) > max_num_examples_per_lang:
            examples_for_lang = random.sample(
                examples_for_lang, max_num_examples_per_lang)

        sampled_examples += [(x, lang) for x in examples_for_lang]
    test_data = sampled_examples

    return test_data

