from RAGModel_Base.ImportsForRag import *
from dotenv import load_dotenv
import os
from pathlib import Path

# Load the .env file from parent directory
dotenv_path = Path(__file__).resolve().parent.parent / "keys.env"
load_dotenv(dotenv_path=dotenv_path)

# Now access your keys
hf_token = os.getenv("hf_Token")
google_token = os.getenv("GOOGLE_API_KEY")


class Search_Semantic:
    def __init__(self, query):
        self.pages_and_chunks = None
        self.embeddings = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model_id = "google/gemma-7b-it"
        self.result_text = self.Semantic_Rag_DotProduct_Search(query)

    def Semantic_Rag_DotProduct_Search(self, query):
        with open("text_chunks_and_embeddings.pkl", "rb") as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x)
        self.pages_and_chunks = df.to_dict(orient="records")
        self.embeddings = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(self.device)
        return self.SearchQuery(query)

    def SearchQuery(self, query):
        results_text = None
        query = query.lower()
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)
        start_time = timer()
        dot_scores = util.dot_score(a=query_embedding, b=self.embeddings)[0]
        end_time = timer()

        print(f"Time take to get scores: {end_time - start_time:.5f} seconds.")
        top_results = torch.topk(dot_scores, k=5)

        # ✅ Make sure to initialize results_text as an empty string
        results_text = f"Query: '{query}'\nTop Results (retrieved in {end_time - start_time:.2f}s):\n\n"

        for score, idx in zip(top_results[0], top_results[1]):
            chunk = self.pages_and_chunks[idx]["sentence_chunk"]
            page = self.pages_and_chunks[idx]["page_number"]
            results_text += f"📝 **Score**: {score:.4f}\n📄 **Page**: {page}\n\n{chunk}\n\n{'-' * 40}\n"

        self.result_text = results_text
        return results_text

        # print(f"Query: '{query}'\nResults:")
        # # for score, idx in zip(top_results[0], top_results[1]):
        # #     print(f"Score: {score:.4f}")
        # #     print("Text:")
        # #     self.print_wrapped(self.pages_and_chunks[idx]["sentence_chunk"])
        # #     print(f"Page number: {self.pages_and_chunks[idx]['page_number']}\n")
        # for score, idx in zip(top_results[0], top_results[1]):
        #     chunk = self.pages_and_chunks[idx]["sentence_chunk"]
        #     page = self.pages_and_chunks[idx]["page_number"]
        #     results_text += f"📝 **Score**: {score:.4f}\n📄 **Page**: {page}\n\n{chunk}\n\n{'-' * 40}\n"
        #
        # self.result_text = results_text
        # return results_text

    def print_wrapped(self, text, width=100):
        from textwrap import fill
        print(fill(text, width=width))


class LLmSearch:
    # Class-level variables to hold the model and tokenizer
    llm_model = None
    tokenizer = None

    def __init__(self, query):
        self.pages_and_chunks = None
        self.embeddings = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model_id = "google/gemma-7b-it"
        self.use_quantization_config = True

        # Load model only once (if not already loaded)
        self._load_model()

        # Run inference
        self.result_text = self.Semantic_Rag_DotProduct_Search(query)

    def _load_model(self):
        if LLmSearch.tokenizer is None or LLmSearch.llm_model is None:
            print("🔄 Loading LLM model and tokenizer...")
            attn_implementation = (
                "flash_attention_2" if is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8 else "sdpa"
            )

            LLmSearch.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token)
            LLmSearch.llm_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                quantization_config=self.quantization_config if self.use_quantization_config else None,
                low_cpu_mem_usage=False,
                attn_implementation=attn_implementation,
                token=hf_token
            )

            if not self.use_quantization_config:
                LLmSearch.llm_model.to("cuda")
            print("✅ Model and tokenizer loaded.")

    def Semantic_Rag_DotProduct_Search(self, query):
        with open("text_chunks_and_embeddings.pkl", "rb") as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x)
        self.pages_and_chunks = df.to_dict(orient="records")
        self.embeddings = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(self.device)
        return self.LocalLLM(query)

    def LocalLLM(self, query):
        tokenizer = LLmSearch.tokenizer
        model = LLmSearch.llm_model

        dialogue_template = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(dialogue_template, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = model.generate(**input_ids, max_new_tokens=2048)
        outputs_decoded = tokenizer.decode(outputs[0])
        response = outputs_decoded.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')
        return response.strip()


# class LLmSearch:
#     def __init__(self, query):
#         self.pages_and_chunks = None
#         self.embeddings = None
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.embedding_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
#         self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
#         self.model_id = "google/gemma-7b-it"
#         self.use_quantization_config = True
#         self.result_text = self.Semantic_Rag_DotProduct_Search(query)
#
#     def Semantic_Rag_DotProduct_Search(self, query):
#         with open("text_chunks_and_embeddings.pkl", "rb") as f:
#             data = pickle.load(f)
#         df = pd.DataFrame(data)
#         df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x)
#         self.pages_and_chunks = df.to_dict(orient="records")
#         self.embeddings = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(self.device)
#         return self.LocalLLM(query)
#
#     def get_model_num_params(self, model):
#         return sum([param.numel() for param in model.parameters()])
#
#     def get_model_mem_size(self, model):
#         mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
#         mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
#         mem_bytes = mem_params + mem_buffers
#         return {
#             "model_mem_bytes": mem_bytes,
#             "model_mem_mb": round(mem_bytes / (1024 ** 2), 2),
#             "model_mem_gb": round(mem_bytes / (1024 ** 3), 2)
#         }
#
#     def LocalLLM(self, query):
#         attn_implementation = (
#             "flash_attention_2" if is_flash_attn_2_available() and torch.cuda.get_device_capability(0)[0] >= 8 else "sdpa"
#         )
#
#         tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token)
#         self.llm_model = AutoModelForCausalLM.from_pretrained(
#             self.model_id,
#             torch_dtype=torch.float16,
#             quantization_config=self.quantization_config if self.use_quantization_config else None,
#             low_cpu_mem_usage=False,
#             attn_implementation=attn_implementation,
#             token=hf_token
#         )
#
#         if not self.use_quantization_config:
#             self.llm_model.to("cuda")
#
#         dialogue_template = [{"role": "user", "content": query}]
#         prompt = tokenizer.apply_chat_template(dialogue_template, tokenize=False, add_generation_prompt=True)
#         input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
#
#         outputs = self.llm_model.generate(**input_ids, max_new_tokens=256)
#         outputs_decoded = tokenizer.decode(outputs[0])
#         response = outputs_decoded.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')
#         print(f"Output text:\n{response.strip()}")
#         return response

