from dotenv import load_dotenv

load_dotenv()

import os
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getlogger("llama_index").addHandler(logging.StreamHandler(stream=sys.stdout))

OpenAILike