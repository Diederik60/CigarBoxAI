import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()
data_path = os.getenv('DATA_PATH')
if data_path:
    os.chdir(data_path)
else:
    print("ERROR: DATA_PATH not set in .env!")
    exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: Install with: pip install google-generativeai")
    sys.exit(1)

class SimpleGeminiCaller:
    """Simple Gemini Flash 2.5 caller with prompt templates"""
    
    def __init__(self, data_file: str, prompts_dir: str):
        self.data_file = Path(data_file)
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Load survey data
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.survey_data = json.load(f)
        
        # Setup API
        api_key = os.environ.get("GEMINI_API_KEY") or input("Enter Gemini API key: ")
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        

    
    def load_prompt(self, prompt_name: str = "default") -> dict:
        """Load prompt template"""
        prompt_file = self.prompts_dir / f"{prompt_name}.json"
        
        if not prompt_file.exists():
            print(f"Prompt '{prompt_name}' not found, using default")
            prompt_file = self.prompts_dir / "default.json"
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def call_gemini(self, question: str, prompt_name: str = "default") -> str:
        """Call Gemini with question and data"""
        prompt_template = self.load_prompt(prompt_name)
        
        # Create full prompt
        full_prompt = f"""{prompt_template['system_prompt']}

SURVEY DATA:
{json.dumps(self.survey_data, indent=2, ensure_ascii=False)}

VRAAG: {question}

ANTWOORD:"""
        
        # Call Gemini
        response = self.model.generate_content(full_prompt)
        return response.text

def main():
    if len(sys.argv) < 2:
        print("Usage: python gemini_caller.py <data_file>")
        print("Example: python gemini_caller.py ./gemini_ready/gemini_optimized_data.json")
        return
    
    data_file = sys.argv[1]
    prompts_dir = "../CigarBoxAI/scripts/prompts"
    prompt_name = "default"
    
    caller = SimpleGeminiCaller(data_file, prompts_dir)
    
    print(f"Ready! Using prompt: {prompt_name}")
    print("Type your questions (or 'quit' to exit):")
    
    while True:
        question = input("\n> ").strip()
        if question.lower() == 'quit':
            break
        
        if question:
            try:
                answer = caller.call_gemini(question, prompt_name)
                print(f"\nAnswer:\n{answer}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()