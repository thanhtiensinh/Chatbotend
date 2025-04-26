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
model = genai.GenerativeModel("gemini-1.0-pro")
chat = model.start_chat(history=[])

# Đếm số lần gọi Gemini API
gemini_call_count = 0
GEMINI_CALL_LIMIT = 50  # Giới hạn 50 lần gọi mỗi ngày (tùy chỉnh theo quota của bạn)

# Thêm bộ nhớ đệm (cache) cho câu trả lời từ Gemini
CACHE_FILE = "gemini_cache.json"
def load_cache():
    return load_json_data(CACHE_FILE) if os.path.exists(CACHE_FILE) else {}

def save_cache(cache):
    save_json_data(CACHE_FILE, cache)

gemini_cache = load_cache()

# Danh sách stop words (các từ không quan trọng)
STOP_WORDS = {"là", "của", "tại", "ở", "và", "thì", "có", "được", "cho", "với", "trong", "vậy", "thế", "nào", "đã"}

# Từ khóa quan trọng liên quan đến thời gian
TIME_KEYWORDS = {"thời gian", "bao giờ", "khi nào", "đã", "chưa"}

# Danh sách câu hỏi không liên quan
IRRELEVANT_QUESTIONS = ["bạn ăn cơm chưa", "bạn khỏe không", "bạn đang làm gì", "bạn có người yêu không"]

# Từ điển đồng nghĩa
SYNONYMS = {
    "tui": "tôi",
    "co": "có"
}

# Chuẩn hóa câu hỏi: loại bỏ dấu câu, chuyển về chữ thường, bỏ stop words
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    # Thay thế từ đồng nghĩa
    for synonym, standard in SYNONYMS.items():
        text = text.replace(synonym, standard)
    words = [word for word in text.split() if word not in STOP_WORDS]
    return ' '.join(words)

# Tìm câu hỏi tương tự dựa trên từ khóa
def find_best_match(question, data, threshold=0.5):
    if not data:
        logger.warning("Không có dữ liệu huấn luyện để tìm kiếm.")
        return None

    # Chuẩn hóa câu hỏi người dùng
    normalized_question = normalize_text(question)
    question_words = set(normalized_question.split())
    
    # Kiểm tra xem câu hỏi có chứa từ khóa thời gian không
    has_time_keywords = bool(question_words.intersection(TIME_KEYWORDS))
    
    # Lấy danh sách câu hỏi từ dữ liệu huấn luyện
    questions = list(data.keys())
    best_match = None
    best_score = 0

    # Duyệt qua từng câu hỏi trong dữ liệu huấn luyện
    for q in questions:
        normalized_q = normalize_text(q)
        q_words = set(normalized_q.split())
        
        # Tính số từ khóa chung
        common_words = question_words.intersection(q_words)
        
        # Tăng trọng số nếu câu hỏi chứa từ khóa thời gian
        q_has_time_keywords = bool(q_words.intersection(TIME_KEYWORDS))
        if has_time_keywords and q_has_time_keywords:
            score = len(common_words) / max(len(question_words), len(q_words), 1) * 1.5
        else:
            score = len(common_words) / max(len(question_words), len(q_words), 1)
        
        # Ghi log để kiểm tra
        logger.debug(f"So sánh: '{normalized_question}' với '{normalized_q}' -> Score: {score}, Common words: {common_words}")
        
        if score >= threshold and score > best_score:
            best_score = score
            best_match = q

    # Nếu không tìm thấy, thử dùng difflib
    if not best_match:
        normalized_questions = [normalize_text(q) for q in questions]
        matches = difflib.get_close_matches(normalized_question, normalized_questions, n=1, cutoff=threshold)
        if matches:
            match_idx = normalized_questions.index(matches[0])
            best_match = questions[match_idx]
            logger.debug(f"Dùng difflib: Tìm thấy khớp với '{best_match}'")

    return best_match

# Gọi API Gemini
def ask_gemini_v2(question):
    global gemini_call_count
    if question in gemini_cache:
        logger.info(f"Đã tìm thấy câu trả lời trong cache cho câu hỏi: {question}")
        return gemini_cache[question]

    if gemini_call_count >= GEMINI_CALL_LIMIT:
        logger.warning("Đã đạt giới hạn số lần gọi Gemini API trong ngày.")
        return "Xin lỗi, tôi đã đạt giới hạn trả lời hôm nay. Bạn có thể hỏi lại vào ngày mai hoặc hỏi các câu về thông tin tuyển sinh, học phí, hoặc chương trình đào tạo của HPU nhé!"

    try:
        start_time = time.time()
        response = chat.send_message(question)  # Loại bỏ tham số timeout
        end_time = time.time()
        logger.info(f"Thời gian gọi Gemini API: {end_time - start_time:.2f} giây")
        answer = response.text.strip() if response else "Không có phản hồi từ Gemini."
        gemini_cache[question] = answer
        save_cache(gemini_cache)  # Lưu cache vào file
        gemini_call_count += 1
        return answer
    except Exception as e:
        logger.error(f"Lỗi khi gọi Gemini API: {str(e)}")
        return f"Lỗi khi gọi Gemini API: {str(e)}"

# Lưu lịch sử chat
def save_chat_history(user_question, bot_response):
    history = load_json_data(CHAT_HISTORY_FILE)
    history.append({"question": user_question, "answer": bot_response})
    save_json_data(CHAT_HISTORY_FILE, history)

# Đọc dữ liệu từ file JSON
def load_json_data(filepath):
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Tệp {filepath} không tồn tại.")
            return []
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Lỗi khi đọc tệp {filepath}: {e}")
        return []

# Lưu dữ liệu vào file JSON
def save_json_data(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except IOError as e:
        logger.error(f"Lỗi khi lưu tệp {filepath}: {e}")

# Đọc dữ liệu huấn luyện
def load_trained_data(filepath):
    try:
        if not os.path.exists(filepath):
            logger.error(f"Tệp {filepath} không tồn tại.")
            return {}
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, dict) and "câu hỏi" in data:
                question_list = data["câu hỏi"]
            elif isinstance(data, list):
                question_list = data
            else:
                logger.error(f"Kỳ vọng một danh sách hoặc từ điển với khóa 'câu hỏi', nhưng nhận được {type(data)}")
                return {}
            if not isinstance(question_list, list):
                logger.error(f"Giá trị của 'câu hỏi' phải là một danh sách, nhưng nhận được {type(question_list)}")
                return {}
            trained_dict = {}
            for item in question_list:
                if not isinstance(item, dict):
                    logger.warning(f"Bỏ qua mục không hợp lệ: {item} (không phải từ điển)")
                    continue
                if "Bạn" not in item or "HPU" not in item:
                    logger.warning(f"Bỏ qua mục thiếu khóa: {item}")
                    continue
                trained_dict[item["Bạn"]] = item["HPU"]
            logger.info(f"Đã tải dữ liệu huấn luyện với {len(trained_dict)} câu hỏi.")
            return trained_dict
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi khi đọc tệp trained_data: {e}")
        return {}
    except Exception as e:
        logger.error(f"Lỗi không mong muốn khi đọc trained_data: {e}")
        return {}

# Lấy câu trả lời
def get_answer(question, data):
    question = question.lower().strip()

    # Kiểm tra lời chào
    greeting_questions = ["alo", "chào", "hello", "hi"]
    if question in greeting_questions:
        return "Chào bạn! Tôi là Chatbot HPU, bạn có câu hỏi gì về trường không? Tôi có thể giúp bạn với thông tin tuyển sinh, học phí, hoặc các chương trình đào tạo."

    # Kiểm tra câu hỏi không liên quan
    if question in IRRELEVANT_QUESTIONS:
        return "Tôi là chatbot, tôi không có người yêu đâu! Bạn có muốn hỏi gì về trường HPU không?"

    # Kiểm tra nếu câu hỏi liên quan đến danh tính của chatbot
    identity_questions = [
        "bạn là ai",
        "bạn là chatbot gì",
        "chatbot này là của ai",
        "tên của bạn là gì",
        "ai tạo ra bạn",
        "bạn là chatbot của ai"
    ]

    for q in identity_questions:
        similarity = difflib.SequenceMatcher(None, question, q).ratio()
        if similarity > 0.8:
            return "Tôi là Chatbot HPU, được tạo ra bởi Đại học Quản lý và Công nghệ Hải Phòng (HPU). Tôi ở đây để giúp bạn giải đáp các thắc mắc về trường, từ thông tin tuyển sinh, học phí, đến các chương trình đào tạo. Bạn muốn hỏi gì?"

    # Tìm trong dữ liệu huấn luyện trước
    best_match = find_best_match(question, data)
    if best_match:
        logger.info(f"Tìm thấy câu hỏi tương tự: {best_match}")
        answer = data.get(best_match)
    else:
        logger.info(f"Không tìm thấy câu hỏi tương giống: {question}, gọi Gemini API.")
        # Gọi Gemini API nếu không tìm thấy trong dữ liệu
        answer = ask_gemini_v2(question)

    save_chat_history(question, answer)
    return answer

# Khởi tạo dữ liệu huấn luyện
CHAT_HISTORY_FILE = "chat_history.json"
trained_data = load_trained_data("trained_data.json")