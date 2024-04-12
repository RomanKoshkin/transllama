TARGET_LANGUAGE = 'Russian'
# MODEL_NAME = 'gpt-4'
# MODEL_NAME = 'gpt-3.5-turbo-0613'
MODEL_NAME = 'gpt-4-1106-preview'
WAIT_K = 1
MAX_TOKENS = 10
TEMPERATURE = 0.0
WAIT_TOKEN = "▁▁"

SYS_PROMPT_INIT = f"You are a professional conference interpreter. Given an English text you translate it into {TARGET_LANGUAGE} as accurately and as concisely as possible, NEVER adding comments of your own. You output translation when the information available in the source is unambiguous, otherwise you output the null token (\"{WAIT_TOKEN}\"), not flanked by anything else. It's important that you get this right."

SYS_PROMPT = f"You are a professional conference interpreter. Given an English text you translate it into {TARGET_LANGUAGE} as accurately and as concisely as possible, NEVER adding comments of your own. You output translation when the information available in the source is unambiguous, otherwise you output the null token (\"{WAIT_TOKEN}\"), not flanked by anything else. It's important that you get this right."

# python se.py     --source SOURCES/mustc2_ende_source_4000_valid.txt     --target OFFLINE_TARGETS/mustc2_ende_target_4000_valid.txt     --agent agent_openai.py     --wait_k 1     --config_id -8220