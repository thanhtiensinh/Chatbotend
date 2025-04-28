import os
import sys
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Thêm import này
from dotenv import load_dotenv
from supabase import create_client, Client
from cbai import load_json_data, save_chat_history, ask_gemini_v2, get_answer, trained_data, CHAT_HISTORY_FILE

# Tắt mã màu trong log của Werkzeug
os.environ["NO_COLOR"] = "1"

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tùy chỉnh logging của Flask/Werkzeug
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.setLevel(logging.INFO)
werkzeug_handler = logging.StreamHandler()
werkzeug_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s'
))
werkzeug_logger.handlers = [werkzeug_handler]

# Thay đổi mã hóa của console thành UTF-8
if sys.platform == "win32":
    os.system("chcp 65001")
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Load biến môi trường từ file .env
load_dotenv(dotenv_path="API.env")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL hoặc SUPABASE_KEY không được tìm thấy trong API.env")

# Khởi tạo Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)  # Thêm dòng này để cho phép CORS từ tất cả domain

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/form")
def form():
    return render_template("api_form.html")

@app.route("/thongtindangky_new")
def thongtindangky():
    return render_template("thongtindangky_new.html")

@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        user_question = request.json.get("question")
        if not user_question:
            return jsonify({"error": "Không có câu hỏi được cung cấp."}), 400
        
        if not trained_data:
            logger.warning("Không có dữ liệu huấn luyện, thử gọi Gemini API...")
            answer = ask_gemini_v2(user_question)
            save_chat_history(user_question, answer)
            return jsonify({"answer": answer})
            
        answer = get_answer(user_question, trained_data)
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Lỗi khi xử lý yêu cầu: {str(e)}")
        return jsonify({"error": f"Lỗi khi xử lý yêu cầu: {str(e)}"}), 500

@app.route("/api/chat-history", methods=["GET"])
def get_chat_history():
    history = load_json_data(CHAT_HISTORY_FILE)
    return jsonify(history)

@app.route("/api/register", methods=["POST"])
def register():
    try:
        user_data = request.json
        logger.info("Dữ liệu nhận được từ frontend: %s", user_data)

        if not isinstance(user_data, dict):
            return jsonify({"error": "Dữ liệu phải là một đối tượng JSON."}), 400

        required_fields = ["name", "phone", "email", "major"]
        missing_fields = [field for field in required_fields if field not in user_data or not user_data[field]]
        if missing_fields:
            return jsonify({"error": f"Thiếu các trường: {', '.join(missing_fields)}"}), 400

        if len(user_data["name"]) > 100 or len(user_data["email"]) > 100 or len(user_data["major"]) > 100:
            return jsonify({"error": "Tên, email hoặc ngành học vượt quá 100 ký tự."}), 400
        if len(user_data["phone"]) > 20:
            return jsonify({"error": "Số điện thoại vượt quá 20 ký tự."}), 400

        data = {
            "name": user_data["name"],
            "phone": user_data["phone"],
            "email": user_data["email"],
            "major": user_data["major"],
            "message": user_data.get("message", None)
        }

        response = supabase.table("consultation_users").insert(data).execute()
        logger.info(f"Đã chèn bản ghi: {response.data}")

        return jsonify({"message": "Đăng ký thành công!"})
    except Exception as e:
        logger.error(f"Lỗi khi xử lý đăng ký: {str(e)}")
        return jsonify({"error": f"Lỗi: {str(e)}"}), 500

@app.route("/api/get-students", methods=["GET"])
def get_students():
    try:
        response = supabase.table("consultation_users").select("*").execute()
        students = response.data
        return jsonify(students)
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách sinh viên: {str(e)}")
        return jsonify({"error": f"Lỗi: {str(e)}"}), 500

@app.route("/api/add-student", methods=["POST"])
def add_student():
    try:
        user_data = request.json
        logger.info("Dữ liệu nhận được để thêm sinh viên: %s", user_data)

        if not isinstance(user_data, dict):
            return jsonify({"error": "Dữ liệu phải là một đối tượng JSON."}), 400

        required_fields = ["name", "phone", "email", "major"]
        missing_fields = [field for field in required_fields if field not in user_data or not user_data[field]]
        if missing_fields:
            return jsonify({"error": f"Thiếu các trường: {', '.join(missing_fields)}"}), 400

        if len(user_data["name"]) > 100 or len(user_data["email"]) > 100 or len(user_data["major"]) > 100:
            return jsonify({"error": "Tên, email hoặc ngành học vượt quá 100 ký tự."}), 400
        if len(user_data["phone"]) > 20:
            return jsonify({"error": "Số điện thoại vượt quá 20 ký tự."}), 400

        data = {
            "name": user_data["name"],
            "phone": user_data["phone"],
            "email": user_data["email"],
            "major": user_data["major"],
            "message": user_data.get("message", None)
        }

        response = supabase.table("consultation_users").insert(data).execute()
        logger.info(f"Đã chèn bản ghi: {response.data}")

        return jsonify({"message": "Thêm sinh viên thành công!"})
    except Exception as e:
        logger.error(f"Lỗi khi thêm sinh viên: {str(e)}")
        return jsonify({"error": f"Lỗi: {str(e)}"}), 500

@app.route("/api/update-student", methods=["POST"])
def update_student():
    try:
        user_data = request.json
        logger.info("Dữ liệu nhận được để cập nhật sinh viên: %s", user_data)

        if not isinstance(user_data, dict):
            return jsonify({"error": "Dữ liệu phải là một đối tượng JSON."}), 400

        required_fields = ["id", "name", "phone", "email", "major"]
        missing_fields = [field for field in required_fields if field not in user_data or not user_data[field]]
        if missing_fields:
            return jsonify({"error": f"Thiếu các trường: {', '.join(missing_fields)}"}), 400

        if len(user_data["name"]) > 100 or len(user_data["email"]) > 100 or len(user_data["major"]) > 100:
            return jsonify({"error": "Tên, email hoặc ngành học vượt quá 100 ký tự."}), 400
        if len(user_data["phone"]) > 20:
            return jsonify({"error": "Số điện thoại vượt quá 20 ký tự."}), 400

        data = {
            "name": user_data["name"],
            "phone": user_data["phone"],
            "email": user_data["email"],
            "major": user_data["major"],
            "message": user_data.get("message", None)
        }

        response = supabase.table("consultation_users").update(data).eq("id", user_data["id"]).execute()
        logger.info(f"Đã cập nhật bản ghi: {response.data}")

        if not response.data:
            return jsonify({"error": "Không có bản ghi nào được cập nhật."}), 500

        return jsonify({"message": "Cập nhật sinh viên thành công!"})
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật sinh viên: {str(e)}")
        return jsonify({"error": f"Lỗi: {str(e)}"}), 500

@app.route("/api/delete-student", methods=["POST"])
def delete_student():
    try:
        user_data = request.json
        logger.info("Dữ liệu nhận được để xóa sinh viên: %s", user_data)

        if not isinstance(user_data, dict) or "id" not in user_data:
            return jsonify({"error": "Thiếu trường id."}), 400

        response = supabase.table("consultation_users").delete().eq("id", user_data["id"]).execute()
        logger.info(f"Đã xóa bản ghi: {response.data}")

        if not response.data:
            return jsonify({"error": "Không có bản ghi nào được xóa."}), 500

        return jsonify({"message": "Xóa sinh viên thành công!"})
    except Exception as e:
        logger.error(f"Lỗi khi xóa sinh viên: {str(e)}")
        return jsonify({"error": f"Lỗi: {str(e)}"}), 500
