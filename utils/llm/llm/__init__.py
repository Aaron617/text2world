from .openai_gpt import OPENAI_GPT
import pdb
from .azure_gpt import OPENAI_GPT_AZURE
from .claude import CLAUDE
from .vllm import VLLM
from common.registry import registry
from .huggingface import HgModels
from .msal_gpt import MSAL_GPT
from .deepseek_r1 import DeepSeek_R1
from .o3_mini import O3_MINI
__all__ = [
    "OPENAI_GPT",
    "OPENAI_GPT_AZURE",
    "VLLM",
    "CLAUDE",
    "HgModels",
    "MSAL_GPT",
    "DeepSeek_R1",
    "O3_MINI"
]


def load_llm(name, config):
    llm = registry.get_llm_class(name).from_config(config)
    
    return llm
    
