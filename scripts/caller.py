import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import webbrowser
import tempfile
from datetime import datetime

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
        
        # Store conversation history for HTML output
        self.conversation_history = []
    
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
    
    def format_terminal_output(self, question: str, answer: str) -> str:
        """Format output for better terminal readability"""
        # Terminal colors
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        CYAN = '\033[96m'
        MAGENTA = '\033[95m'
        END = '\033[0m'
        
        # Format the answer with better structure
        formatted_answer = self.format_answer_content(answer)
        
        # Create formatted output
        output = f"""
{BLUE}{'='*70}{END}
{BOLD}{UNDERLINE}VRAAG:{END} {question}
{BLUE}{'='*70}{END}

{GREEN}{BOLD}ANTWOORD:{END}
{YELLOW}{'─'*70}{END}
{formatted_answer}
{YELLOW}{'─'*70}{END}
"""
        return output
    
    def format_answer_content(self, answer: str) -> str:
        """Format the answer content with better headings and structure"""
        # Terminal colors
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        CYAN = '\033[96m'
        MAGENTA = '\033[95m'
        END = '\033[0m'
        
        lines = answer.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines but preserve spacing
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Format main headings (lines that end with colon and are standalone)
            if stripped.endswith(':') and len(stripped) > 10 and '**' not in stripped:
                formatted_lines.append(f"{CYAN}{BOLD}{UNDERLINE}{stripped}{END}")
                continue
            
            # Format bold sections **text**
            if '**' in stripped:
                # Replace **text** with colored bold text
                import re
                formatted_line = re.sub(r'\*\*(.*?)\*\*', f'{MAGENTA}{BOLD}\\1{END}', stripped)
                formatted_lines.append(formatted_line)
                continue
            
            # Format bullet points
            if stripped.startswith('*'):
                formatted_lines.append(f"  {GREEN}•{END} {stripped[1:].strip()}")
                continue
            
            # Format numbered items or percentages
            if any(char.isdigit() for char in stripped[:10]) and ('%' in stripped or stripped[0].isdigit()):
                formatted_lines.append(f"  {YELLOW}▶{END} {stripped}")
                continue
            
            # Format conclusion or important statements
            if any(keyword in stripped.lower() for keyword in ['conclusie', 'samenvatting', 'belangrijkste', 'opvallend']):
                formatted_lines.append(f"{RED}{BOLD}🔍 {stripped}{END}")
                continue
            
            # Regular text
            formatted_lines.append(stripped)
        
        return '\n'.join(formatted_lines)
    
    def generate_html_report(self) -> str:
        """Generate HTML report of the conversation"""
        html_template = f"""
<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CigarBoxAI Survey Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .conversation {{
            background: white;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .question {{
            background: #e3f2fd;
            padding: 20px;
            border-left: 4px solid #2196f3;
        }}
        .question h3 {{
            margin: 0 0 10px 0;
            color: #1976d2;
        }}
        .answer {{
            padding: 25px;
            line-height: 1.6;
        }}
        .answer h4 {{
            color: #4caf50;
            margin-top: 0;
        }}
        .timestamp {{
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
        }}
        .summary {{
            background: #fff3e0;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 4px solid #ff9800;
        }}
        .no-conversations {{
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 50px;
        }}
        pre {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre-wrap;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .stat {{
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.2);
            border-radius: 8px;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CigarBoxAI Survey Analysis</h1>
        <p>Intelligente analyse van je project data</p>
        <div class="stats">
            <div class="stat">
                <span class="stat-number">{len(self.conversation_history)}</span>
                <span>Vragen gesteld</span>
            </div>
            <div class="stat">
                <span class="stat-number">{datetime.now().strftime('%H:%M')}</span>
                <span>Laatst bijgewerkt</span>
            </div>
        </div>
    </div>
    
    <div class="summary">
        <h2>Sessie Overzicht</h2>
        <p><strong>Data bestand:</strong> {self.data_file.name}</p>
        <p><strong>Aantal survey responses:</strong> {len(self.survey_data) if isinstance(self.survey_data, list) else 'Onbekend'}</p>
        <p><strong>Gegenereerd op:</strong> {datetime.now().strftime('%d-%m-%Y om %H:%M:%S')}</p>
    </div>
"""
        
        if self.conversation_history:
            for i, (question, answer, timestamp) in enumerate(self.conversation_history, 1):
                html_template += f"""
    <div class="conversation">
        <div class="question">
            <h3>Vraag {i}:</h3>
            <p>{question}</p>
            <div class="timestamp">{timestamp}</div>
        </div>
        <div class="answer">
            <h4>Antwoord:</h4>
            <pre>{answer}</pre>
        </div>
    </div>
"""
        else:
            html_template += """
    <div class="no-conversations">
        <p>Nog geen vragen gesteld in deze sessie.</p>
    </div>
"""
        
        html_template += """
    <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
        <p>Gegenereerd door CigarBoxAI • Voor meer informatie over je project data</p>
    </div>
</body>
</html>
"""
        return html_template
    
    def save_and_open_html(self):
        """Save HTML report and open in browser"""
        try:
            print(f"\nGenerating HTML rapport...")
            html_content = self.generate_html_report()
            print(f"HTML content generated: {len(html_content)} characters")
            
            # Create reports directory structure
            reports_dir = Path("Rotterdam Bluegrass Festival/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to reports directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_file = reports_dir / f"cigarbox_report_{timestamp}.html"
            
            print(f"Saving to: {html_file}")
            print(f"Current directory: {os.getcwd()}")
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Check if file was created
            if html_file.exists():
                file_size = html_file.stat().st_size
                print(f"File created successfully: {html_file} ({file_size} bytes)")
            else:
                print(f"ERROR: File was not created!")
                return
            
            # Try to open in browser (works better in WSL)
            try:
                # For WSL, try different approaches
                if os.path.exists('/mnt/c'):  # WSL detected
                    print("WSL detected, converting path...")
                    
                    # Get absolute path and convert to WSL format for browser
                    abs_path = html_file.resolve()
                    wsl_path = f"\\\\wsl.localhost\\Ubuntu-22.04{abs_path}"
                    
                    print(f"WSL path: {wsl_path}")
                    
                    # Try to open with Windows default browser using WSL path
                    import subprocess
                    try:
                        # Use explorer.exe to open the file with default browser
                        subprocess.run(['explorer.exe', str(abs_path)], check=True)
                        print(f"HTML rapport geopend in je browser!")
                    except:
                        # Fallback: try cmd.exe with different approach
                        try:
                            subprocess.run(['cmd.exe', '/c', 'start', '', f'"{wsl_path}"'], check=True)
                            print(f"HTML rapport geopend in je browser!")
                        except:
                            print(f"Automatisch openen mislukt. Open handmatig:")
                            print(f"In Windows Explorer: {wsl_path}")
                            print(f"Of type in Windows: \\\\wsl.localhost\\Ubuntu-22.04{abs_path}")
                    
                    print(f"Bestand opgeslagen als: {html_file}")
                    print(f"WSL pad: {wsl_path}")
                else:
                    # Regular Linux/Mac
                    webbrowser.open(f'file://{html_file.resolve()}')
                    print(f"HTML rapport geopend in je browser!")
                    print(f"Bestand opgeslagen als: {html_file.resolve()}")
                    
            except Exception as e:
                print(f"HTML rapport gegenereerd!")
                print(f"Bestand opgeslagen als: {html_file}")
                print(f"Open het bestand handmatig in je browser.")
                print(f"Error bij openen: {e}")
                
        except Exception as e:
            print(f"ERROR creating HTML file: {e}")
            import traceback
            traceback.print_exc()

def main():
    if len(sys.argv) < 2:
        print("Usage: python gemini_caller.py <data_file>")
        print("Example: python gemini_caller.py ./gemini_ready/gemini_optimized_data.json")
        return
    
    data_file = sys.argv[1]
    prompts_dir = "../CigarBoxAI/scripts/prompts"
    prompt_name = "default"
    
    caller = SimpleGeminiCaller(data_file, prompts_dir)
    
    # Welcome message
    print(f"""
CigarBoxAI Survey Analyzer
{'='*50}
Data bestand: {data_file}
AI Model: Gemini 2.5 Flash
Prompt: {prompt_name}
{'='*50}

Commando's:
   - Stel je vraag over de survey data
   - 'html' - Genereer en open HTML rapport
   - 'quit' - Afsluiten

Typ je vraag hieronder:
""")
    
    while True:
        question = input(f"\n> ").strip()
        
        if question.lower() == 'quit':
            print("\nTot ziens!")
            break
        
        if question.lower() == 'html':
            caller.save_and_open_html()
            continue
        
        if question:
            try:
                print(f"\nVerwerking vraag...")
                answer = caller.call_gemini(question, prompt_name)
                
                # Store in conversation history
                timestamp = datetime.now().strftime('%H:%M:%S')
                caller.conversation_history.append((question, answer, timestamp))
                
                # Display formatted output
                formatted_output = caller.format_terminal_output(question, answer)
                print(formatted_output)
                
            except Exception as e:
                print(f"\nError: {e}")

if __name__ == "__main__":
    main()