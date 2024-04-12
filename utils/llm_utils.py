import bitsandbytes as bnb
import re, regex
from functools import reduce


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def contains_japanese(text):
    # Define a regular expression pattern for Japanese characters using Unicode character property escapes
    japanese_pattern = regex.compile(r'([\p{IsHan}\p{IsBopo}\p{IsHira}\p{IsKatakana}]+)', regex.UNICODE)

    # Use the regular expression to search for Japanese characters in the text
    return bool(japanese_pattern.search(text))


def remove_parentheses(s):
    return re.sub(r'\(.*?\)', '', s)  # remove parentheses and anytning they flank


def remove_multiple_spaces(s):
    return re.sub(r'\s+', ' ', s)  # multiple spaces


def fix_spaces_around_punctuation(input_string):
    input_string = re.sub(r'\s*,\s*', ', ', input_string)
    input_string = re.sub(r'\s*;\s*', '; ', input_string)
    input_string = re.sub(r'\s*:\s*', ': ', input_string)
    return re.sub(r'\s*\.\s*', '. ', input_string)


def remove_extra_punctuation(s):
    s = re.sub(r'\.{2,}', '', s)  # 2 or more consecutive periods
    return re.sub(r'!', '.', s)


def remove_roles(s):
    return re.sub(r'^.{0, 12}: ', '', s)  # if a line starts with MIKE: I don't know... up to 12 chars in NAME


def remove_extra_punctuation_ja(s):
    return re.sub(r'\．{2,}', '', s)  # 2 or more consecutive periods


def add_period_if_missing_ja(s):
    if not re.search(r'\。$', s):
        s += '。'
    return s


def cleanup(d, min_num_words_in_source=9, max_num_words_in_source=70):
    func_list = [remove_parentheses, remove_multiple_spaces, remove_extra_punctuation, remove_roles]
    d['en'] = reduce(lambda val, func: func(val), func_list, d['en'])  # apply a pipeline of transforms
    d['ru'] = reduce(lambda val, func: func(val), func_list, d['ru'])  # apply a pipeline of transforms

    # d['en'] = remove_roles(remove_extra_punctuation(remove_multiple_spaces(remove_parentheses(d['en'])))).strip()
    # d['ru'] = remove_roles(remove_extra_punctuation(remove_multiple_spaces(remove_parentheses(d['ru'])))).strip()

    # drop items that are more than one sentence
    if (d['en'].count(".") > 1) or (d['ru'].count(".") > 1):
        return None
    if (":" in d['en']) or (":" in d['ru']):  # reject ones with colons
        return None
    if ("." in d['en'][:-2]) or ("." in d['ru'][:-2]):  # drop multiple-sentence examples
        return None
    if len(d['en'].strip().split(' ')) < min_num_words_in_source:  # reject too short sentences
        return None
    if len(d['en'].strip().split(' ')) > max_num_words_in_source:  # reject too short sentences
        return None
    else:
        return d


def replace_all_but_last_period(d: dict) -> dict:
    """ To convert multiple Japanese sentences into one by replacing all but the last period with commas """
    s = d['en'].strip()
    t = d['ru'].strip()
    clean_t = re.sub("。", "、", t)
    clean_t = re.sub(r'、$', '。', clean_t)
    return dict(en=s.strip(), ru=clean_t.strip())


# def replace_all_but_last_period_in_string(s: str) -> str:
#     """ To convert multiple Japanese sentences into one by replacing all but the last period with commas """
#     s = s.strip()
#     clean_s = re.sub("。", "、", s)
#     clean_s = re.sub(r'、$', '。', clean_s)
#     return clean_s


def join_punctuation_ja(lst):
    processed = []
    for word in lst:
        if word in ['。', '、']:
            processed[-1] += word
        else:
            processed.append(word)
    return processed


def add_period_if_missing_ja(s):
    if not re.search(r'\。$', s):
        s += '。'
    return s


def replace_period_except_at_end_ja(s):
    return re.sub(r"。(?!$)", "、", s)


def replace_space_with_comman_except_at_end_ja(s):
    return s.strip().replace(" ", "、")


def cleanup_ja(d):
    func_list = []
    func_list_ja = [
        remove_parentheses,
        remove_multiple_spaces,
        remove_roles,
        replace_space_with_comman_except_at_end_ja,
        replace_period_except_at_end_ja,
        add_period_if_missing_ja,
    ]

    d['en'] = reduce(lambda val, func: func(val), func_list, d['en'])  # apply a pipeline of transforms
    d['ru'] = reduce(lambda val, func: func(val), func_list_ja, d['ru'])  # apply a pipeline of transforms

    # drop items that are more than one sentence
    if (d['en'].count(".") > 1) or (d['ru'].count(".") > 1):
        return None
    if (":" in d['en']) or (":" in d['ru']):  # reject ones with colons
        return None
    if ("." in d['en'][:-2]) or ("." in d['ru'][:-2]):  # drop multiple-sentence examples
        return None
    if (len(d['en']) < 100) or (len(d['ru']) < 100):  # reject too short sentences
        return None
    else:
        return d