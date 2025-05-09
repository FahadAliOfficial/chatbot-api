from flask import Flask, request, jsonify
from llama_index.core import (
    SimpleDirectoryReader, Settings, VectorStoreIndex, 
    StorageContext, load_index_from_storage
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import google.generativeai as genai
from supabase import create_client
from dotenv import load_dotenv
from flask_cors import CORS
import os, shutil, tempfile

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500", "http://localhost:3000"])

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)
BUCKET_NAME = "vector-dbs"
LOCAL_VECTOR_DIR = "./VectorStore"

# ─── CONFIGURE GEMINI ───────────────────────────────────────────────────────────
def configure_settings(api):
    llm = Gemini(model="models/gemini-2.0-flash-exp", api_key=api)
    embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=api)
    genai.configure(api_key=api)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 800
    Settings.chunk_overlap = 20

# ─── BUILD VECTOR DB ────────────────────────────────────────────────────────────
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
        print("Data Loaded")
        file_prefix = f"{course_id}/files"
        files = supabase.storage.from_("course-materials").list(file_prefix)
        if not files or len(files) == 0:
            return jsonify({"error": "No files found"}), 404
        print("Files retrieved")

        with tempfile.TemporaryDirectory() as tmpdir:
            for f in files:
                path = f"{file_prefix}/{f['name']}"
                content = supabase.storage.from_("course-materials").download(path)
                with open(os.path.join(tmpdir, f['name']), "wb") as out_file:
                    out_file.write(content)
                print("Files Downloaded")

            configure_settings(api)
            reader = SimpleDirectoryReader(tmpdir)
            docs = reader.load_data()
            index = VectorStoreIndex.from_documents(docs)
            print("DB - Created")

            # Step 4: Save vector DB locally
            local_vector_path = os.path.join(tmpdir, "vector_index")
            print(local_vector_path)
            index.storage_context.persist(local_vector_path)
            print("Saved Locally")

            # Step 5: Upload to Supabase bucket
            # Step 5: Upload to Supabase bucket
            upload_prefix = f"{course_id}/"
            storage = supabase.storage.from_(BUCKET_NAME)

            # 🔁 Delete all files in course_id directory once before uploading
            try:
                files = storage.list(f"{course_id}/")
                file_paths = [f"{course_id}/{file['name']}" for file in files]
                print(f"Deleting files: {file_paths}")

                if file_paths:
                    response = storage.remove(file_paths)
                    print(f"Delete response: {response}")
                else:
                    print("No files found to delete.")
            except Exception as e:
                print(f"Warning: Could not delete files for course '{course_id}' - {str(e)}")

            # ⬆ Do this before the loop
            print("Uploading to supabase")

            # ✅ Upload new files
            for root, _, files in os.walk(local_vector_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, local_vector_path)
                    supabase_path = f"{upload_prefix}/{relative_path}"

                    with open(full_path, "rb") as file_data:
                        storage.upload(supabase_path, file_data)



            print("done")

        return jsonify({"message": f"Vector DB built and uploaded to Supabase at path '{upload_prefix}'."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── EXPAND QUERY ───────────────────────────────────────────────────────────────
def expand_query(query):
    synonyms = {
        "example": ["sample", "instance"],
        "query": ["question", "inquiry"]
    }
    expanded = query
    for word, syns in synonyms.items():
        for syn in syns:
            expanded += f" OR {syn}"
    return expanded

# ─── LOAD VECTOR DB + RAG ────────────────────────────────────────────────────────
@app.route("/load_vector_db", methods=["POST"])
def load_vector_db():
    data = request.json
    api_key, course_id, raw_query = data.get("api_key"), data.get("course_id"), data.get("query")
    query = expand_query(raw_query)

    if not (api_key and course_id and raw_query):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        configure_settings(api_key)

        # Reset local dir
        if os.path.exists(LOCAL_VECTOR_DIR):
            shutil.rmtree(LOCAL_VECTOR_DIR)
        os.makedirs(LOCAL_VECTOR_DIR, exist_ok=True)

        # Download from Supabase
        vector_prefix = f"{course_id}/"
        files = supabase.storage.from_(BUCKET_NAME).list(vector_prefix)
        if not files:
            return jsonify({"error": "No vector DB found for course."}), 404

        for file in files:
            content = supabase.storage.from_(BUCKET_NAME).download(f"{vector_prefix}/{file['name']}")
            local_path = os.path.join(LOCAL_VECTOR_DIR, file['name'])
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(content)

        # Load vector index
        storage_context = StorageContext.from_defaults(persist_dir=LOCAL_VECTOR_DIR)
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query(query)

        raw_answer = response.response.strip()
        top_node = response.source_nodes[0] if response.source_nodes else None
        # score = getattr(top_node, "score", 1.0) if top_node else 0.0
        score = getattr(top_node, "score", 1.0)
        print("response generation")
        print("raw",raw_answer)
        # Decide path: Fallback vs Refine
        if not raw_answer or score < 0.3:
            print("[Fallback] No relevant vector DB answer.")
            fallback_llm = Gemini(model="models/gemini-2.0-flash-exp", api_key=api_key)
            fallback = fallback_llm.complete(raw_query).text.strip()
            print(f"fallback: {fallback}")

            return jsonify({
                "message": "No vector DB match. Answer generated using Gemini.",
                "response": fallback
            })

        # Refine with Gemini
        refine_llm = Gemini(model="models/gemini-2.0-flash-exp", api_key=api_key)
        refined_prompt = f"Improve and simplify this for a student:\n\n{raw_answer}"
        refined = refine_llm.complete(refined_prompt).text.strip()
        print(f"refined: {refined}")
        return jsonify({
            "message": "Answer matched in vector DB and refined.",
            "response": refined
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)