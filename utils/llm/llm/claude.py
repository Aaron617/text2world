from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os
import time
from common.registry import registry
import pdb
import tiktoken
import requests
import json


@registry.register_llm("claude")
class CLAUDE:
    def __init__(self,
                 engine="claude-2",
                 temperature=0,
                 max_tokens=200,
                 top_p=1,
                 stop=None,
                 retry_delays=60,  # in seconds
                 max_retry_iters=5,
                 system_message='',
                 context_length=100000,
                 split=None
                 ):
        # self.url = "https://api.anthropic.com/v1/complete"
        # self.base_url = "http://172.17.0.1:18829"  # for request
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.retry_delays = retry_delays
        self.max_retry_iters = max_retry_iters
        self.system_message = system_message
        self.context_length = context_length
        self.xml_split = split
        self.anthropic = None
        return

    def llm_inference_ours(self, messages):
        Baseurl = os.environ.get("OPENAI_API_BASE")
        Skey = os.environ.get("OPENAI_API_KEY")
        payload = json.dumps({
            "model": self.engine,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": messages
                }
            ]
        })
        url = Baseurl
        # url = Baseurl + "/chat/completions"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {Skey}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)

        data = response.json()

        content = data
        return content['choices'][0]['message']['content']

    def llm_inference(self, messages):
        try:
            return self.llm_inference_ours(messages)
        except:
            response = self.anthropic.completions.create(
                model=self.engine,
                max_tokens_to_sample=self.max_tokens,
                prompt=messages,
                stop_sequences=self.stop,
                temperature=self.temperature
            )
            return response.completion

    def generate(self, system_message, prompt):
        extra_prompt = " \n\nAssistant:Do you only want to output the next Action? \n\nHuman: Yes, please do. "
        prompt = HUMAN_PROMPT + system_message + prompt + extra_prompt + AI_PROMPT
        for attempt in range(self.max_retry_iters):
            try:
                return True, self.llm_inference(prompt)  # return success, completion
            except Exception as e:
                print(f"Error on attempt {attempt + 1}")
                if attempt < self.max_retry_iters - 1:  # If not the last attempt
                    time.sleep(self.retry_delays)  # Wait before retrying

                else:
                    print("Failed to get completion after multiple attempts.")
                    raise e

        return False, None

    def num_tokens_from_messages(self, messages, model="claude-2"):
        """Return the number of tokens used by a list of messages."""
        default_tokens_fixed = self.default_tokens_fixed
        if "claude-2" in model:
            default_tokens_fixed = default_tokens_fixed  # tokens of HUMAN_PROMPT and AI_PROMPT
        elif "claude-instant-1" in model:
            default_tokens_fixed = default_tokens_fixed
        else:
            raise NotImplementedError(
                f"""Not implemented for model {model}."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += self.anthropic.count_tokens(message["content"])
        num_tokens += default_tokens_fixed
        return num_tokens

    @classmethod
    def from_config(cls, config):

        engine = config.get("engine", "claude-2")
        temperature = config.get("temperature", 0)
        max_tokens = config.get("max_tokens", 100)
        system_message = config.get("system_message", "You are a helpful assistant.")
        top_p = config.get("top_p", 1)
        stop = config.get("stop", ["\n\nHuman:"])
        retry_delays = config.get("retry_delays", 10)
        max_retry_iters = config.get("max_retry_iters", 15)
        context_length = config.get("context_length", 100000)
        xml_split = config.get("xml_split", {"example": ["<example>", "</example>"]})
        return cls(engine=engine,
                   temperature=temperature,
                   max_tokens=max_tokens,
                   top_p=top_p,
                   stop=stop,
                   retry_delays=retry_delays,
                   max_retry_iters=max_retry_iters,
                   system_message=system_message,
                   context_length=context_length,
                   split=xml_split)
