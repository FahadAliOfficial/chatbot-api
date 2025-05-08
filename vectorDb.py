from flask import Flask, request, jsonify
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import google.generativeai as genai
from supabase import create_client
import os
import shutil
import tempfile  # ✅ Missing import
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:3000"])
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

# Create the Supabase client using the environment variables
supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)

BUCKET_NAME = "vector-dbs"
LOCAL_VECTOR_DIR = "./VectorStore"

def configure_settings(api):
    """Configure Gemini LLM and embedding model for llama-index."""
    llm = Gemini(model="models/gemini-2.0-flash-exp", api_key=api)
    embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=api)
    genai.configure(api_key=api)
    Settings.llm = Gemini(api_key=api)
    Settings.embed_model = GeminiEmbedding(api_key=api)

    Settings.chunk_size = 800
    Settings.chunk_overlap = 20

@app.route("/build_vector_db", methods=["POST"])
def build_vector_db():
    try:
        data = request.get_json()
        course_id = data.get("course_id")
        api = data.get("api")
        if not course_id:
            return jsonify({"error": "Missing course_id"}), 400
        if not api:
            return jsonify({"error": "Missing api"}), 400

        # Step 1: List files from the course bucket path
        file_prefix = f"{course_id}/files"
        files = supabase.storage.from_("course-materials").list(file_prefix)
        if not files or len(files) == 0:
            return jsonify({"error": "No files found"}), 404

        # Step 2: Download files locally
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in files:
                path = f"{file_prefix}/{f['name']}"
                content = supabase.storage.from_("course-materials").download(path)
                with open(os.path.join(tmpdir, f['name']), "wb") as out_file:
                    out_file.write(content)
            
            # Status Message: Files downloaded successfully
            print("Step 2 completed: Files downloaded locally.")

            # Step 3: Build Vector DB
            configure_settings(api)  # ✅ Use helper function
            reader = SimpleDirectoryReader(tmpdir)
            docs = reader.load_data()
            index = VectorStoreIndex.from_documents(docs)

            # Status Message: Vector DB built successfully
            print("Step 3 completed: Vector DB built.")

            # Step 4: Save locally
            index.storage_context.persist(f"./V")
            print(f"Step 4 completed: Vector DB persisted for course {course_id}")

        return jsonify({"message": "Vector DB built and saved successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/load_vector_db", methods=["POST"])
def load_vector_db():
    data = request.json
    api_key = data.get("api_key")
    if not api_key:
        return jsonify({"error": "Missing API key"}), 400

    try:
        # Status Message: Configuring models
        print("Step 1: Configuring Gemini model...")
        configure_settings(api_key)

        # Clear existing local directory
        if os.path.exists(LOCAL_VECTOR_DIR):
            shutil.rmtree(LOCAL_VECTOR_DIR)
        os.makedirs(LOCAL_VECTOR_DIR, exist_ok=True)

        # Status Message: Local directory cleared
        print("Step 2: Local directory cleared.")

        # Download files from Supabase bucket
        files = supabase.storage.from_(BUCKET_NAME).list()
        if not files or len(files) == 0:
            return jsonify({"error": "No vector DB files found in Supabase bucket"}), 404

        for file in files:
            file_path = file["name"]
            local_path = os.path.join(LOCAL_VECTOR_DIR, file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            content = supabase.storage.from_(BUCKET_NAME).download(file_path)
            with open(local_path, "wb") as f:
                f.write(content)

        # Status Message: Files downloaded from Supabase
        print("Step 3: Files downloaded from Supabase.")

        # Load vector index
        storage_context = StorageContext.from_defaults(persist_dir=LOCAL_VECTOR_DIR)
        index = load_index_from_storage(storage_context)

        # Status Message: Vector DB loaded
        print("Step 4: Vector DB loaded from local storage.")

        # Test query
        query_engine = index.as_query_engine()
        test_response = query_engine.query("Hello, can you confirm vector DB is loaded?")

        return jsonify({
            "message": "Vector DB loaded successfully from Supabase.",
            "test_response": test_response.response
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

