from flask import Flask, request, jsonify
from deepface import DeepFace
from flask_cors import CORS
import os
import cv2
import base64
import numpy as np
import tempfile
import pyodbc
from numpy.linalg import norm

app = Flask(__name__)
CORS(app)

# ============================================================
# SQL SERVER CONNECTION
# ============================================================
CONN_STR = (
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=DESKTOP-8G6MFH8\\MSSQLSERVER01;"
    "Database=HRMS;"
    "Trusted_Connection=yes;"
)

# ----------------------------------------------------
# Helper: Decode base64 → OpenCV image
# ----------------------------------------------------
def decode_base64_image(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("❌ Failed to decode base64 image:", str(e))
        return None


# ============================================================
# 1️⃣ GET EMBEDDING
# ============================================================
@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    try:
        data = request.get_json()
        print("📥 Received data keys:", list(data.keys()))

        image_path = data.get("image_path")
        image_base64 = data.get("image_base64")

        if not image_path and not image_base64:
            return jsonify({"error": "Missing 'image_path' or 'image_base64'"}), 400

        if image_base64:
            img = decode_base64_image(image_base64)
            if img is None:
                return jsonify({"error": "Failed to decode base64 image"}), 400
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, img)
                image_path = temp_path
        elif image_path:
            image_path = os.path.normpath(image_path)
            if not os.path.exists(image_path):
                return jsonify({"error": f"Image not found: {image_path}"}), 400

        print(f"✅ Processing image: {image_path}")

        embedding_obj = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            enforce_detection=False
        )
        embedding = embedding_obj[0]['embedding']
        print(f"✅ Embedding generated successfully. Length: {len(embedding)}")

        return jsonify({
            "embedding": embedding,
            "message": "Embedding generated successfully"
        })

    except Exception as e:
        print("❌ Error while generating embedding:", str(e))
        return jsonify({"error": str(e)}), 500


# ============================================================
# 2️⃣ LIVENESS + SQL EMBEDDING COMPARISON + EMPLOYEE NAME
# ============================================================
@app.route('/liveness', methods=['POST'])
def liveness_check():
    try:
        data = request.get_json()
        print("📥 Liveness request keys:", list(data.keys()))

        image_base64 = data.get("image_base64")
        if not image_base64:
            return jsonify({"error": "Missing 'image_base64'"}), 400

        # Decode base64 → temp image
        img = decode_base64_image(image_base64)
        if img is None:
            return jsonify({"error": "Failed to decode base64 image"}), 400

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, img)

        print(f"✅ Performing liveness + embedding for: {temp_path}")

        # ----------------------------------------------------
        # Liveness detection using DeepFace emotion analysis
        # ----------------------------------------------------
        analysis = DeepFace.analyze(img_path=temp_path, actions=['emotion'], enforce_detection=False)
        is_live = bool("dominant_emotion" in analysis[0] or "dominant_emotion" in analysis)

        # ----------------------------------------------------
        # Generate embedding
        # ----------------------------------------------------
        current_embedding_obj = DeepFace.represent(img_path=temp_path, model_name="ArcFace", enforce_detection=False)
        current_embedding = np.array(current_embedding_obj[0]['embedding'])

        # ----------------------------------------------------
        # Connect to SQL Server
        # ----------------------------------------------------
        print("🖥️ Connecting to SQL Server...")
        conn = pyodbc.connect(CONN_STR)
        print("✅ SQL connection established.")
        cursor = conn.cursor()

        # Join query to get FullName
        query = """
        SELECT emb.EMP_Info_Id, emb.Embedding, emp.FullName
        FROM EMP_FaceEmbeddings AS emb
        INNER JOIN EMP_Info AS emp ON emb.EMP_Info_Id = emp.Id
        WHERE emb.Embedding IS NOT NULL AND emp.IsActive = 1
        """
        cursor.execute(query)

        best_match = None
        best_name = None
        best_similarity = -1.0
        threshold = 0.45  # ArcFace cosine similarity threshold

        for row in cursor.fetchall():
            emp_id = str(row[0])
            db_embedding_str = row[1]
            emp_name = row[2]

            if not db_embedding_str or not db_embedding_str.startswith('['):
                continue

            try:
                db_embedding = np.array(list(map(float, db_embedding_str.strip('[]').split(','))))
            except:
                continue

            # Cosine similarity (higher = better)
            similarity = np.dot(current_embedding, db_embedding) / (norm(current_embedding) * norm(db_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = emp_id
                best_name = emp_name

        conn.close()

        # Determine match
        is_matched = bool(best_similarity >= threshold)

        result = {
            "is_live": bool(is_live),
            "is_matched": bool(is_matched),
            "best_match_emp_id": best_match if is_matched else None,
            "employee_name": best_name if is_matched else None,
            "match_score": round(float(best_similarity), 4),
            "embedding_length": int(len(current_embedding))
        }

        print("✅ Liveness + match result:", result)
        return jsonify(result)

    except Exception as e:
        print("❌ Error in liveness:", str(e))
        return jsonify({"error": str(e)}), 500


# ============================================================
# 3️⃣ RUN FLASK APP
# ============================================================
if __name__ == '__main__':
    print("🚀 Starting Flask Face Verification API...")
    try:
        conn = pyodbc.connect(CONN_STR)
        print("✅ Initial SQL Server connection test successful.")
        conn.close()
    except Exception as e:
        print("❌ Failed to connect to SQL Server on startup:", str(e))

    app.run(host='0.0.0.0', port=5000, debug=True)
