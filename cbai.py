import json
import os
import difflib
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import re
from collections import Counter
import time

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load biến môi trường từ file .env
load_dotenv(dotenv_path="API.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY không được tìm thấy trong API.env")

# Cấu hình Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Khởi tạo mô hình và phiên chat một lần duy nhất
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])

# Đếm số lần gọi Gemini API
gemini_call_count = 0
GEMINI_CALL_LIMIT = 50

# Danh sách stop words và các hằng số khác
STOP_WORDS = {"là", "của", "tại", "ở", "và", "thì", "có", "được", "cho", "với", "trong", "vậy", "thế", "nào", "đã"}
TIME_KEYWORDS = {"thời gian", "bao giờ", "khi nào", "đã", "chưa"}
IRRELEVANT_QUESTIONS = ["bạn ăn cơm chưa", "bạn khỏe không", "bạn đang làm gì", "bạn có người yêu không"]
SYNONYMS = {"tui": "tôi", "co": "có"}
HPU_KEYWORDS = [
    "học phí", "tuyển sinh", "ngành học", "khoa", "học bổng",
    "HPU", "trường", "chuyên ngành", "xét tuyển", "điểm chuẩn",
    "ký túc", "cơ sở vật chất", "giảng viên", "lịch học"
]

# JSON utilities
def load_json_data(filepath):
    try:
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except:
        return []

def save_json_data(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except:
        pass

# Cache
CACHE_FILE = "gemini_cache.json"
def load_cache():
    return load_json_data(CACHE_FILE) if os.path.exists(CACHE_FILE) else {}

def save_cache(cache):
    save_json_data(CACHE_FILE, cache)

gemini_cache = load_cache()

# Xử lý ngôn ngữ

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    for synonym, standard in SYNONYMS.items():
        text = text.replace(synonym, standard)
    words = [word for word in text.split() if word not in STOP_WORDS]
    return ' '.join(words)

def find_best_match(question, data, threshold=0.5):
    if not data:
        logger.warning("Không có dữ liệu huấn luyện để tìm kiếm.")
        return None
    normalized_question = normalize_text(question)
    question_words = set(normalized_question.split())
    has_time_keywords = bool(question_words.intersection(TIME_KEYWORDS))
    questions = list(data.keys())
    best_match = None
    best_score = 0
    for q in questions:
        normalized_q = normalize_text(q)
        q_words = set(normalized_q.split())
        common_words = question_words.intersection(q_words)
        q_has_time_keywords = bool(q_words.intersection(TIME_KEYWORDS))
        score = len(common_words) / max(len(question_words), len(q_words), 1)
        if has_time_keywords and q_has_time_keywords:
            score *= 1.5
        if score >= threshold and score > best_score:
            best_score = score
            best_match = q
    if not best_match:
        normalized_questions = [normalize_text(q) for q in questions]
        matches = difflib.get_close_matches(normalized_question, normalized_questions, n=1, cutoff=threshold)
        if matches:
            match_idx = normalized_questions.index(matches[0])
            best_match = questions[match_idx]
    return best_match

def build_chat_history_for_gemini(max_turns=5):
    history = load_json_data("chat_history.json")[-max_turns:]  # lấy 5 lượt cuối cùng
    messages = []
    for item in history:
        messages.append({"role": "user", "parts": [item["question"]]})
        messages.append({"role": "model", "parts": [item["answer"]]})
    return messages
def ask_gemini_with_context(conversation_history, current_question):
    global gemini_call_count

    # Tạo prompt có ngữ cảnh từ hội thoại trước
    prompt = (
        "Bạn là chatbot tư vấn tuyển sinh HPU. Dưới đây là hội thoại trước đó:\n"
        + "\n".join([f"Người dùng: {q}\nChatbot: {a}" for q, a in conversation_history[-3:]])
        + f"\nNgười dùng: {current_question}\nChatbot:"
    )

    if current_question in gemini_cache:
        return gemini_cache[current_question]

    if gemini_call_count >= GEMINI_CALL_LIMIT:
        return "Xin lỗi, tôi đã đạt giới hạn trả lời hôm nay."

    try:
        start_time = time.time()
        response = model.generate_content(prompt)
        end_time = time.time()
        logger.info(f"Thời gian gọi Gemini API: {end_time - start_time:.2f} giây")

        answer = response.text.strip() if response else "Không có phản hồi từ Gemini."
        gemini_cache[current_question] = answer
        save_cache(gemini_cache)
        gemini_call_count += 1
        return answer

    except Exception as e:
        return f"Lỗi khi gọi Gemini API (with context): {str(e)}"

def ask_gemini_v2(question):
    global gemini_call_count
    if question in gemini_cache:
        return gemini_cache[question]

    if gemini_call_count >= GEMINI_CALL_LIMIT:
        return "Xin lỗi, tôi đã đạt giới hạn trả lời hôm nay."

    try:
        # Thêm ngữ cảnh mặc định
        context_prefix = (
            "Bạn là Chatbot của Trường Đại học Quản lý và Công nghệ Hải Phòng (HPU). "
            "Nếu người dùng hỏi về 'trường', 'ngành', 'sinh viên', v.v., mặc định đó là HPU.\n\n"
        )
        full_prompt = context_prefix + "Câu hỏi: " + question

        # Khởi tạo chat với history
        history = build_chat_history_for_gemini()
        chat_with_history = model.start_chat(history=history)

        response = chat_with_history.send_message(full_prompt)
        answer = response.text.strip() if response else "Không có phản hồi từ Gemini."

        gemini_cache[question] = answer
        save_cache(gemini_cache)
        gemini_call_count += 1
        return answer

    except Exception as e:
        return f"Lỗi khi gọi Gemini API: {str(e)}"

    except Exception as e:
        return f"Lỗi khi gọi Gemini API: {str(e)}"

def save_chat_history(user_question, bot_response):
    history = load_json_data(CHAT_HISTORY_FILE)
    history.append({"question": user_question, "answer": bot_response})
    save_json_data(CHAT_HISTORY_FILE, history)

def load_trained_data(filepath):
    try:
        if not os.path.exists(filepath):
            return {}
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, dict) and "câu hỏi" in data:
                question_list = data["câu hỏi"]
            elif isinstance(data, list):
                question_list = data
            else:
                return {}
            trained_dict = {}
            for item in question_list:
                if not isinstance(item, dict):
                    continue
                if "Bạn" not in item or "HPU" not in item:
                    continue
                trained_dict[item["Bạn"]] = item["HPU"]
            return trained_dict
    except:
        return {}

def should_use_trained_data(question: str) -> bool:
    question = question.lower()
    for keyword in HPU_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", question):
            return True
    return False

def get_answer(question, data):
    question = question.lower().strip()
    if question in ["alo", "chào", "hello", "hi"]:
        return "Chào bạn! Tôi là Chatbot HPU."
    if question in IRRELEVANT_QUESTIONS:
        return "Tôi là chatbot, tôi không có người yêu đâu!"
    identity_questions = [
        "bạn là ai", "bạn là chatbot gì", "chatbot này là của ai",
        "tên của bạn là gì", "ai tạo ra bạn", "bạn là chatbot của ai"
    ]
    for q in identity_questions:
        if difflib.SequenceMatcher(None, question, q).ratio() > 0.8:
            return "Tôi là Chatbot HPU, được tạo bởi Đại học HPU."

    if should_use_trained_data(question):
        best_match = find_best_match(question, data)
        if best_match:
            answer = data.get(best_match)
        else:
            answer = ask_gemini_v2(question)
    else:
        answer = ask_gemini_v2(question)

    save_chat_history(question, answer)
    return answer

CHAT_HISTORY_FILE = "chat_history.json"
trained_data = load_trained_data("trained_data.json")
