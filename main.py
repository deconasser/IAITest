from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import handle_question
import uvicorn

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Định nghĩa schema đầu vào cho API
class QuestionRequest(BaseModel):
    question: str

# Định nghĩa endpoint chính để xử lý câu hỏi
@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    user_question = request.question

    try:
        # Gọi hàm xử lý câu hỏi
        answer = handle_question(user_question)
        return {"answer": answer}
    except Exception as e:
        # Xử lý lỗi và trả về thông báo lỗi
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Khởi chạy server
if __name__ == "__main__":
    # Thay đổi port nếu cần, ví dụ từ 8080 thành 8000 hoặc một cổng khác
    uvicorn.run(app, host="127.0.0.1", port=8000)
