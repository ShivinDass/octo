def build_prompts(final_example_text, few_shot_texts, separator, y,
                  post_fmt_text, pre_fmt_text, tokenizer, max_length, rng):
    def build_for_prompts(prompts):
        prompt = pre_fmt_text
        for _, p in enumerate(prompts):
            prompt += f"{p}{separator}"

        prompt += final_example_text
        prompt += post_fmt_text
        prompt += y
        return prompt

    def token_length_for(prompts):
        prompt = build_for_prompts(prompts)
        encoded = tokenizer.encode(prompt)
        if 'input_ids' in encoded and isinstance(encoded, dict):
            encoded = encoded['input_ids']

        if len(encoded) == 1:
            encoded = encoded[0]

        return len(encoded)

    fss_in_order = sorted(few_shot_texts, key=lambda x: -len(x))
    num_to_remove = 0
    while True:
        num_tokens = token_length_for(fss_in_order[num_to_remove:])
        if num_tokens <= max_length - 10:
            break

        num_to_remove += 1
        # print('> num_to_remove:', num_to_remove)
        if num_to_remove == len(fss_in_order):
            break

    fss_keep = fss_in_order[num_to_remove:]
    randomized_order = rng.permutation(len(fss_keep))
    final_prompt = build_for_prompts([fss_keep[i] for i in randomized_order],
                                     final_example_text)
    return final_prompt
