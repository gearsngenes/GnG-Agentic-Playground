from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from helpers import UPLOAD_FOLDER, get_embedding
import os
import shutil
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TABLE_OF_CONTENTS_INDEX = "table-of-contents"

class PineconeManager:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.ensure_upload_folder()
        self.ensure_table_of_contents_index()
        self.prefix = "kb-"

    def ensure_upload_folder(self):
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)

    def ensure_table_of_contents_index(self):
        if TABLE_OF_CONTENTS_INDEX not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=TABLE_OF_CONTENTS_INDEX,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

    def list_indexes(self):
        return [idx for idx in self.pc.list_indexes().names() if idx != TABLE_OF_CONTENTS_INDEX and self.prefix in idx]

    def create_index(self, index_name):
        self.pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        self.create_topic_directory(index_name)

    def delete_index(self, index_name):
        self.pc.delete_index(index_name)
        self.delete_topic_directory(index_name)
        toc_index = self.pc.Index(TABLE_OF_CONTENTS_INDEX)
        toc_index.delete(ids=[index_name])

    def create_topic_directory(self, index_name):
        topic_dir = os.path.join(UPLOAD_FOLDER, index_name)
        os.makedirs(topic_dir, exist_ok=True)

    def delete_topic_directory(self, index_name):
        topic_dir = os.path.join(UPLOAD_FOLDER, index_name)
        if os.path.exists(topic_dir):
            shutil.rmtree(topic_dir)

    def upsert_metadata(self, index_name, description):
        index = self.pc.Index(TABLE_OF_CONTENTS_INDEX)
        embedding = get_embedding(description)
        vector = {
            "id": index_name,
            "values": embedding,
            "metadata": {"description": description}
        }
        index.upsert(vectors=[vector])

    def get_index_description(self, index_name):
        toc_index = self.pc.Index(TABLE_OF_CONTENTS_INDEX)
        response = toc_index.fetch(ids=[index_name])
        if response and hasattr(response, "vectors") and index_name in response.vectors:
            metadata = response.vectors[index_name].get("metadata", {})
            return metadata.get("description", "No description available.")
        return "No description available."

    def get_descriptions(self):
        """Retrieve descriptions for all indexes except the table of contents."""
        toc_index = self.pc.Index(TABLE_OF_CONTENTS_INDEX)
        all_indexes = self.list_indexes()
        descriptions = {}
        response = toc_index.fetch(ids=all_indexes)
        if hasattr(response, "vectors"):
            for idx in all_indexes:
                metadata = response.vectors.get(idx, {}).get("metadata", {})
                descriptions[idx] = metadata.get("description", "No description available.")
        return descriptions

    def delete_vectors_by_source(self, index_name, file_name):
        index = self.pc.Index(index_name)
        query_result = index.query(
            vector=[0] * 1536,
            namespace="docs",
            top_k=1000,
            filter={"source": {"$eq": file_name}}
        )
        chunk_ids = [match["id"] for match in query_result.get("matches", [])]
        if chunk_ids:
            index.delete(ids=chunk_ids, namespace="docs")

    def is_embedded(self, index_name, file_name, namespace="docs"):
        """
        Searches all non-table-of-contents indexes for the given file name.
        Returns the index name where the file is embedded, or None if not found.
        """
        index = self.pc.Index(index_name)
        try:
            query_result = index.query(
                vector=[0] * 1536,
                namespace=namespace,
                top_k=1,
                filter={"source": {"$eq": file_name}},
                include_metadata=True
            )
            return True if query_result.get("matches") else False
        except Exception as e:
            print(f"Error checking embedding status of {file_name} in {index_name}")
            return False

    def upsert_vectors(self, index_name, src_doc, file_paths, chunks, embed_type, namespace="docs"):
        pc_vectors = [
            {
                "id": f"{src_doc}-{embed_type}-{i}",
                "values": get_embedding(chunk),
                "metadata": {
                    "content": chunk,
                    "source": src_doc,
                    "file_path": file_paths[i],
                    "type": embed_type
                }
            }
            for i, chunk in enumerate(chunks)
        ]
        index = self.pc.Index(index_name)
        index.upsert(vectors=pc_vectors, namespace=namespace)

    def query_at_index(self, index_name, query, top_k=5):
        """Queries the specified index using the embedded query and returns list of metadata contents."""
        embedding = get_embedding(query)
        index = self.pc.Index(index_name)
        results = index.query(
            vector=embedding,
            top_k=top_k,
            namespace="docs",
            include_metadata=True
        )
        return [match.get("metadata", {}) for match in results.get("matches", [])]

vector_store_manager = PineconeManager()