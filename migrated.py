import chromadb
chroma_client=chromadb.Client()
collection=chroma_client.create_collection(name="new_collections")

collection.add(
    documents=["This is a document","Another doc"],
    metadatas=[{"source":"mysource"},{"source":"mysource"}],
    ids=["id1","id2"]
)

results= collection.query(
    query_texts=["This is a document"],
    n_results=2
)
print(results)