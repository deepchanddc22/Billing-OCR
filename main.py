from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import os
from io import BytesIO
from langchain_community.llms import Ollama
import json
import re

app = FastAPI()

# Initialize the PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize the Ollama model
llm = Ollama(model="gemma2:2b")

def extract_text_from_image(image_bytes: BytesIO) -> str:
    with open("temp_image.jpg", "wb") as temp_file:
        temp_file.write(image_bytes.read())
    result = ocr.ocr("temp_image.jpg", cls=True)
    extracted_text = ''
    for line in result:
        for text_info in line:
            extracted_text += text_info[1][0] + '\n'
    os.remove("temp_image.jpg")  # clean up the temporary image
    return extracted_text

def extract_text_from_pdf(pdf_bytes: BytesIO) -> str:
    images = convert_from_path(BytesIO(pdf_bytes.read()))
    extracted_text = ''
    for image in images:
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path, 'JPEG')
        extracted_text += extract_text_from_image(BytesIO(open(temp_image_path, 'rb').read()))
        os.remove(temp_image_path)  # clean up the temporary image
    return extracted_text

def format_text_with_llm(text: str) -> str:
    format_template = '''json
    {
      "items": [
        {"item": "", "price": float},
        {"item": "", "price": float}
        .
        .
        .and more
      ]
    }
    '''
    prompt = f"use this template exactly {format_template} and from {text} extract item and its price as json"
    response = llm.invoke(prompt, temperature=0.0)
    print(response)
    
    # Extract JSON data from the response
    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```|```s*(\{.*?\})\s*```|```JSON\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_string = json_match.group(1) or json_match.group(2) or json_match.group(3)
            data = json.loads(json_string)
            # print(data)
            return data
        else:
            return {"error": "No valid JSON found in the response"}
    except json.JSONDecodeError as e:
        return {"error": f"JSONDecodeError: {e}"}

@app.post("/extract_text/")
async def extract_text(file: UploadFile = File(...)):
    file_type = file.content_type
    if file_type.startswith("image/"):
        image_bytes = BytesIO(await file.read())
        text = extract_text_from_image(image_bytes)
    elif file_type == "application/pdf":
        pdf_bytes = BytesIO(await file.read())
        text = extract_text_from_pdf(pdf_bytes)
    else:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)
    
    formatted_data = format_text_with_llm(text)
    return JSONResponse(content=formatted_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
