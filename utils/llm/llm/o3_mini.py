import openai
import os
import time
from common.registry import registry
import pdb
import tiktoken

@registry.register_llm("o3_mini")
class O3_MINI:
    def __init__(self,
                 engine="gpt-3.5-turbo-0631",
                 temperature=0,
                 max_completion_tokens=1000,
                 use_azure=True,
                 top_p=1,
                 stop=None,
                 retry_delays=60, # in seconds
                 max_retry_iters=20,
                 context_length=4096,
                 system_message=''
                 ):
        
        self.engine = engine
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.use_azure = use_azure
        self.top_p = top_p
        self.stop = stop
        self.retry_delays = retry_delays
        self.max_retry_iters = max_retry_iters
        self.context_length = context_length
        self.system_message = system_message
        self.usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'prompt_cache_hit_tokens': 0,'prompt_cache_miss_tokens': 0, 'prompt_tokens_details': 0, 'completion_tokens_details': 0}
        self.init_api_key()
        
    def init_api_key(self):
        if self.use_azure:
            openai.api_type = os.environ['OPENAI_API_TYPE']
            openai.api_version = os.environ['OPENAI_API_VERSION']
        else:
            if 'OPENAI_API_KEY' not in os.environ:
                raise Exception("OPENAI_API_KEY environment variable not set.")
            else:
                openai.api_key = os.environ['OPENAI_API_KEY']
            openai.api_base = os.environ['OPENAI_API_BASE'] if 'OPENAI_API_BASE' in os.environ else openai.api_base

    def clear_usage(self):
        for k, v in self.usage.items():
            self.usage[k] = 0

    def update_usage(self, response):
        for k, v in response['usage'].items():
            try:
                self.usage[k] += v
            except:
                self.usage[k] += 0

    def get_usage(self):
        return self.usage

    def llm_inference(self, messages):
        if "o1" in self.engine:
            new_messages = []
            for i in range(len(messages)):
                message = messages[i]
                if message["role"] != "system":
                    new_messages.append(message)
            messages = new_messages
            response = openai.ChatCompletion.create(
                model=self.engine, # engine = "deployment_name".
                messages=messages,
                stop = self.stop,
                # temperature = 1.,
                max_completion_tokens = self.max_tokens,
            )
        else:
            response = openai.ChatCompletion.create(
                model=self.engine, # engine = "deployment_name".
                messages=messages,
                stop = self.stop,
                # temperature = self.temperature,
                max_completion_tokens = self.max_completion_tokens,
            )
        return response['choices'][0]['message']['content']

    def generate(self, system_message, prompt):
        prompt=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        for attempt in range(self.max_retry_iters):
            try:
                return True, self.llm_inference(prompt) # return success, completion
            except Exception as e:
                print(f"Error on attempt {attempt + 1}, {str(e)}")
                if attempt < self.max_retry_iters - 1:  # If not the last attempt
                    time.sleep(self.retry_delays)  # Wait before retrying
                else:
                    print("Failed to get completion after multiple attempts.")
                    # raise e

        return False, None

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        model = self.engine
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens_per_message = 0
        tokens_per_name = 0
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    @classmethod
    def from_config(cls, config):
        
        engine = config.get("engine", "gpt-35-turbo")
        temperature = config.get("temperature", 0)
        max_completion_tokens = config.get("max_completion_tokens", 100)
        system_message = config.get("system_message", "You are a helpful assistant.")
        use_azure = config.get("use_azure", True)
        top_p = config.get("top_p", 1)
        stop = config.get("stop", None)
        retry_delays = config.get("retry_delays", 10)
        context_length = config.get("context_length", 4096)
        return cls(engine=engine,
                   temperature=temperature,
                   max_completion_tokens=max_completion_tokens,
                   use_azure=use_azure,
                   top_p=top_p,
                   retry_delays=retry_delays,
                   system_message=system_message,
                   context_length=context_length,
                   stop=stop)
