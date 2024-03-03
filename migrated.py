import chromadb

chroma_client=chromadb.Client()
embeddings = HuggingFaceEmbeddings()

from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
model.query_instruction = "为这个句子生成表示以用于检索相关文章："


collection=chroma_client.create_collection(name="new_collections")

# collection.add(
#     documents=["This is a document","Another doc"],
#     metadatas=[{"source":"mysource"},{"source":"mysource"}],
#     ids=["id1","id2"]
# )

# results= collection.query(
#     query_texts=["This is a document but also an extension of the stuffs thats happening here."],
#     n_results=2
# )
# print(results)