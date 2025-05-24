from ImportsForRag import *
import os



# Load .env file
load_dotenv(dotenv_path="keys.env")

# Access token
hf_token = os.getenv("HuggingFace_LLM_Token")

class RagModel:
    def __init__(self, pdf_path):
        # os.environ["STREAMLIT_WATCH_DISABLE"] = "true"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Store PDF path and device
        self.pdf_path = pdf_path
        print("*************************************************************************************************")
        print(self.pdf_path)
        self.device = device
        print(device)
        # Placeholder for data
        self.Dataset = pd.DataFrame()
        self.statsDataset = pd.DataFrame()

        # Load embedding model
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)

        # Load quantization config for LLM (Google Gemma)
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        # Model selection (Gemma-7B)
        self.model_id = "google/gemma-7b-it"
        self.use_quantization_config = True


        # Start model workflow
        self.ModelDriver()

    def ModelDriver(self):

        # Reading the dataset
        self.pages_and_text_list = self.ReadingPDF()
        print(self.pages_and_text_list)

        # Convert the pdf to a dataframe object
        self.statsDataset = pd.DataFrame(self.pages_and_text_list)
        print(self.statsDataset.head(10))

        # Sentencizing the Data
        self.Sentencizing_NLP()

        # Chunking
        self.Chunking()

        # Splitting Chunks
        # Splitting each chunk into its own item
        self.SplittingChunks()

        # Run once then comment it out so that we get embeddings saved to a csv file
        self.EmbeddingChunks()

        # Searching for the results of a query in pdf without LLM
        # THis is done by using the dot product between the vecotrs
        # self.Semantic_Rag_DotProduct_Search()


        # Local LLM Model
        # self.LocalLLM()
        os.environ["STREAMLIT_WATCH_DISABLE"] = "true"
        # Local LLM Fallback if query not found in pdf
        # self.PromptFeature_LLM()

    def text_formatter(self, text: str) -> str:
        """Performs minor formatting on text."""
        cleaned_text = text.replace("\n",
                                    " ").strip()  # note: this might be different for each doc (best to experiment)

        # Other potential text formatting functions can go here
        return cleaned_text

    def ReadingPDF(self):
        doc = fitz.open(self.pdf_path)  # open a document
        pages_and_text = []
        for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
            text = page.get_text()  # get plain text encoded as UTF-8
            text = self.text_formatter(text)
            pages_and_text.append(
                {"page_number": page_number - 41,  # adjust page numbers since our PDF starts on page 42
                 "page_char_count": len(text),
                 "page_word_count": len(text.split(" ")),
                 "page_sentence_count_raw": len(text.split(". ")),
                 "page_token_count": len(text) / 4,
                 # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                 "text": text})
        return pages_and_text

    def Sentencizing_NLP(self):
        nlp = English()

        # Add a sentencizer pipeline, see https://spacy.io/api/sentencizer/
        nlp.add_pipe("sentencizer")

        for item in tqdm(self.pages_and_text_list):
            item["sentences"] = list(nlp(item["text"]).sents)

            # Make sure all sentences are strings
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]

            # Count the sentences
            item["page_sentence_count_spacy"] = len(item["sentences"])

        # print(random.sample(self.pages_and_text_list, k=1))
        self.statsDataset = pd.DataFrame(self.pages_and_text_list)
        # print(self.statsDataset.describe().round(2))

    def Chunking(self):
        # Define split size to turn groups of sentences into chunks
        num_sentence_chunk_size = 10

        # Create a function that recursively splits a list into desired sizes
        def split_list(input_list: list,
                       slice_size: int) -> list[list[str]]:
            """
            Splits the input_list into sublists of size slice_size (or as close as possible).

            For example, a list of 17 sentences would be split into two lists of [[10], [7]]
            """
            return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

        # Loop through pages and texts and split sentences into chunks
        for item in tqdm(self.pages_and_text_list):
            item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                                 slice_size=num_sentence_chunk_size)
            item["num_chunks"] = len(item["sentence_chunks"])

        # Sample an example from the group (note: many samples have only 1 chunk as they have <=10 sentences total)

        # Create a DataFrame to get stats
        # print(random.sample(self.pages_and_text_list, k=1))
        self.statsDataset = pd.DataFrame(self.pages_and_text_list)
        # print(self.statsDataset.describe().round(2))

    def SplittingChunks(self):
        # Split each chunk into its own item
        self.pages_and_chunks = []
        for item in tqdm(self.pages_and_text_list):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1',
                                               joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo
                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                # Get stats about the chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 characters

                self.pages_and_chunks.append(chunk_dict)

        # How many chunks do we have?
        # print(len(self.pages_and_chunks))
        # Get stats about our chunks
        self.statsDataset = pd.DataFrame(self.pages_and_chunks)
        # print(self.statsDataset.describe().round(2))

        # Show random chunks with under 30 tokens in length
        min_token_length = 30
        for row in self.statsDataset[self.statsDataset["chunk_token_count"] <=
                                     min_token_length].sample(5).iterrows():
            print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')

        self.pages_and_chunks_over_min_token_len = self.statsDataset[
            self.statsDataset["chunk_token_count"] > min_token_length].to_dict(orient="records")
        # print(self.pages_and_chunks_over_min_token_len[:2])

    def EmbeddingChunks(self):

        # Send the model to the GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # requires a GPU installed, for reference on my local machine, I'm using a NVIDIA RTX 2080
        self.embedding_model.to(device)

        # Create embeddings one by one on the GPU
        for item in tqdm(self.pages_and_chunks_over_min_token_len):
            item["embedding"] = self.embedding_model.encode(item["sentence_chunk"])

        # Turn text chunks into a single list
        text_chunks = [item["sentence_chunk"] for item in self.pages_and_chunks_over_min_token_len]

        # Embed all texts in batches
        # text_chunk_embeddings = self.embedding_model.encode(text_chunks,
        #                                                     batch_size=32,
        #                                                     convert_to_tensor=True)
        # print(text_chunk_embeddings)

        # Embed all texts in batches
        self.embeddings = self.embedding_model.encode(text_chunks,
                                                      batch_size=32,
                                                      convert_to_tensor=True).to(self.device)
        # Save to pickle
        embeddings_save_path = "text_chunks_and_embeddings.pkl"
        with open(embeddings_save_path, "wb") as f:
            pickle.dump(self.pages_and_chunks_over_min_token_len, f)
        print(f"✅ Embeddings saved to {embeddings_save_path}")


