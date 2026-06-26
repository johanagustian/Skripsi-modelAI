import random
from unittest import result

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL_PATH = "./models"

print("Memuat model T5 ke memori, harap tunggu...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
if device == "cpu":
    model = model.float()
print(f"Model berhasil dimuat menggunakan: {device.upper()}")

def parse_t5_output(raw_text):
    try:
        if "; options" in raw_text:
            parts = raw_text.split("; options")
        else:
            parts = raw_text.split(";")
            
        question_text = parts[0].replace("question:", "").strip()
        raw_options_string = parts[1].replace("options", "").strip()
        raw_options = raw_options_string.split("|")
        
        options = []
        correct_answer = ""
        
        for opt in raw_options:
            opt = opt.strip()
            if "true;" in opt.lower() or "true:" in opt.lower() or "true " in opt.lower():
                clean_opt = opt.replace("true;", "").replace("true:", "").replace("true", "").strip()
                options.append(clean_opt)
                correct_answer = clean_opt
            else:
                if opt: 
                    options.append(opt)
                
        if not correct_answer and len(options) > 0:
            correct_answer = options[0]

        # Mengunci urutan asli bawaan model T5 ke variabel baru
        default_options = list(options)

        # Mengacak variabel 'options' asli menggunakan random.shuffle
        if len(options) > 1:
            random.shuffle(options)

        # Variabel return tetap dipertahankan, hanya ditambahkan default_options
        return {
            "question_text": question_text,
            "default_options": default_options,  # Cadangan opsi asli dari model
            "options": options,                  # Opsi yang sudah teracak
            "correct_answer": correct_answer
        }
    except Exception as e:
        fallback_options = ["A", "B", "C", "D"]
        default_fallback = list(fallback_options)
        random.shuffle(fallback_options)
        return {
            "question_text": raw_text,
            "default_options": default_fallback,
            "options": fallback_options,
            "correct_answer": "A"
        }

def generate_adaptive_question(difficulty: str, context: str, max_length: int = 128, mode_kreatif: bool = True):
    prompt = f"generate {difficulty} question: context: {context} "

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    if mode_kreatif:
        gen_args = {
            "do_sample": True,
            "top_k": 40,
            "top_p": 0.9,
            "temperature": 0.3,  
            "repetition_penalty": 1.5,
            "no_repeat_ngram_size": 3
        }
    else:
        gen_args = {
            "num_beams": 4,
            "early_stopping": True,
            "repetition_penalty": 1.2
        }

    # Gunakan torch.no_grad()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            **gen_args
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
   
    return parse_t5_output(result)