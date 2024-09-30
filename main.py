import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Access the OpenAI API key from the environment
openai_api_key = os.getenv('OPENAI_API_KEY')



UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create upload folder if it doesn't exist

# Initialize OpenAI chat-based LLM
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

# Resume parsing prompt template
resume_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Extract the following information from the resume:\n"
             "- Name\n"
             "- Email\n"
             "- Phone Number\n"
             "- Skills\n"
             "- Experience\n\n"
             "Resume Text:\n{text}\n\n"
             "Provide the extracted information in JSON format."
)

# Store parsed resumes in memory (you can also use a database)
parsed_resumes = {}

@app.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    
    # Save the uploaded file
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)

    # Extract text from the PDF and parse it
    pdf_text = extract_text_from_pdf(file_location)
    
    # Use the resume prompt template to create the final prompt
    final_prompt = resume_prompt_template.format(text=pdf_text)

    # Run the LLM with the formatted prompt
    response = llm(final_prompt)  # Call the model with the formatted string
    
    # Store the parsed data with the filename as the key
    parsed_resumes[file.filename] = response.content  # Store the parsed response

    # # Print parsed data to the console
    # print("Parsed data for:", file.filename)
    # print(response.content)

    return {"info": f"File '{file.filename}' uploaded  successfully."}


@app.get("/retrieve_resume/")
async def retrieve_resume(filename: str):
    file_location = os.path.join(UPLOAD_FOLDER, filename)

    # Check if the file exists
    if os.path.exists(file_location):
        pdf_text = extract_text_from_pdf(file_location)

        # Use the resume prompt template to create the final prompt
        final_prompt = resume_prompt_template.format(text=pdf_text)

        # Run the LLM with the formatted prompt
        response = llm(final_prompt)  # Call the model with the formatted string
        
        # Extract the parsed data from the response
        parsed_data = response.content
        
        print(f"Returning parsed data for: {filename}")  # Debug info
        return JSONResponse(content={"filename": filename, "parsed_data": parsed_data})
    else:
        print(f"File not found: {file_location}")  # Debug info
        return JSONResponse(content={"error": "File not found."}, status_code=404)


def extract_text_from_pdf(pdf_file_path):
    text = ''
    with open(pdf_file_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ''
    return text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
