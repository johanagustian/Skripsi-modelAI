from fastapi import FastAPI
from routers import question_router

app = FastAPI(title="AI Model Service", description="API untuk Model T5 IRT")

app.include_router(question_router.router)

@app.get("/")
def root():
    return {"message": "AI Service sedang berjalan!"}