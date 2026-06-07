from fastapi import APIRouter
from schemas.question_schema import GenerateRequest
from services.t5_service import generate_adaptive_question

router = APIRouter()

@router.post("/generate")
def generate_question(req: GenerateRequest):
    result = generate_adaptive_question(req.difficulty, req.context, req.mode_kreatif)
    return result