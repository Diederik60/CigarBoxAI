import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class GeminiFlashDataOptimizer:
    """
    Optimizes comprehensive survey data for Google Gemini Flash 2.5 consumption
    Creates focused, contextual data structures that work best with Gemini's capabilities
    """
    
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load comprehensive data
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
    
    def extract_project_context(self) -> Dict[str, Any]:
        """Extract essential project context for Gemini"""
        project_data = self.raw_data.get("project_data", {})
        current_project = project_data.get("current_project", {})
        responses_info = self.raw_data.get("responses_data", {}).get("file_info", {})
        
        return {
            "project_name": current_project.get("name", "Unknown Project"),
            "project_type": "Cultural Event Survey",
            "survey_period": current_project.get("date", "Unknown"),
            "total_respondents": responses_info.get("total_respondents", 0),
            "data_collection_method": "CigarBox survey platform",
            "project_description": current_project.get("description", "Survey data analysis for cultural event"),
            "confidence_score": project_data.get("project_identification", {}).get("match_confidence", 0)
        }
    
    def create_structured_demographics(self) -> Dict[str, Any]:
        """Create clean demographic breakdown optimized for Gemini"""
        column_analysis = self.raw_data.get("responses_data", {}).get("column_analysis", {})
        
        demographics = {}
        
        # Map common demographic fields
        demographic_mappings = {
            "gender": ["gender", "geslacht", "ben je"],
            "age": ["age", "leeftijd", "wat is je leeftijd"],
            "education": ["education", "opleiding", "hoogst voltooide"],
            "location": ["location", "locatie", "waar woon je"],
            "cultural_background": ["cultural", "culturele achtergrond", "culturele"],
            "company": ["gezelschap", "met wie", "company"]
        }
        
        for demo_key, possible_columns in demographic_mappings.items():
            for col_name, col_data in column_analysis.items():
                if any(term in col_name.lower() for term in possible_columns):
                    if "value_distribution" in col_data:
                        demographics[demo_key] = {
                            "question": col_name,
                            "responses": col_data["value_distribution"],
                            "percentages": col_data.get("value_percentages", {}),
                            "completion_rate": f"{col_data.get('completion_rate', 0):.1f}%"
                        }
                    break
        
        return demographics
    
    def create_satisfaction_metrics(self) -> Dict[str, Any]:
        """Extract and organize satisfaction/rating data for Gemini"""
        column_analysis = self.raw_data.get("responses_data", {}).get("column_analysis", {})
        
        satisfaction = {}
        
        # Look for rating/satisfaction columns
        rating_keywords = ["rating", "waardering", "waardeer", "tevredenheid", "satisfaction"]
        
        for col_name, col_data in column_analysis.items():
            if any(keyword in col_name.lower() for keyword in rating_keywords):
                if "statistics" in col_data:
                    stats = col_data["statistics"]
                    satisfaction[col_name] = {
                        "average": stats.get("mean"),
                        "median": stats.get("median"),
                        "scale": f"{stats.get('min', 0)}-{stats.get('max', 10)}",
                        "response_count": col_data.get("non_null_count", 0),
                        "completion_rate": f"{col_data.get('completion_rate', 0):.1f}%"
                    }
                elif "value_distribution" in col_data:
                    satisfaction[col_name] = {
                        "distribution": col_data["value_distribution"],
                        "percentages": col_data.get("value_percentages", {}),
                        "response_count": col_data.get("non_null_count", 0)
                    }
        
        return satisfaction
    
    def create_behavioral_insights(self) -> Dict[str, Any]:
        """Extract behavioral data (motivations, discovery methods, etc.)"""
        column_analysis = self.raw_data.get("responses_data", {}).get("column_analysis", {})
        
        behavioral = {}
        
        # Common behavioral question patterns
        behavioral_patterns = {
            "visit_motivation": ["reden", "waarom", "motivatie", "om welke reden"],
            "discovery_method": ["bekend geraakt", "hoe ben je", "ontdekt", "heard about"],
            "transportation": ["vervoer", "transport", "hoe ben je gekomen"],
            "accompaniment": ["gezelschap", "met wie", "accompanied"],
            "repeat_visit": ["eerder", "previous", "first time", "vaker"]
        }
        
        for behavior_key, keywords in behavioral_patterns.items():
            for col_name, col_data in column_analysis.items():
                if any(keyword in col_name.lower() for keyword in keywords):
                    if "value_distribution" in col_data:
                        behavioral[behavior_key] = {
                            "question": col_name,
                            "top_responses": dict(list(col_data["value_distribution"].items())[:5]),
                            "percentages": dict(list(col_data.get("value_percentages", {}).items())[:5]),
                            "total_responses": col_data.get("non_null_count", 0)
                        }
                    break
        
        return behavioral
    
    def extract_key_pdf_insights(self) -> List[str]:
        """Extract key insights from PDF for Gemini context"""
        pdf_analysis = self.raw_data.get("pdf_analysis", {})
        insights = []
        
        # Get percentages with context
        percentages = pdf_analysis.get("extracted_metrics", {}).get("percentages", [])
        if percentages:
            insights.append(f"Key percentages found in report: {', '.join(map(str, percentages[:10]))}")
        
        # Get important numbers
        numbers = pdf_analysis.get("extracted_metrics", {}).get("numbers", [])
        if numbers:
            insights.append(f"Important figures: {', '.join(numbers[:5])}")
        
        # Get sections
        sections = pdf_analysis.get("identified_sections", [])
        if sections:
            insights.append(f"Report covers: {', '.join(sections[:8])}")
        
        return insights
    
    def create_gemini_optimized_data(self) -> Dict[str, Any]:
        """Create the main Gemini-optimized data structure"""
        
        project_context = self.extract_project_context()
        demographics = self.create_structured_demographics()
        satisfaction = self.create_satisfaction_metrics()
        behavioral = self.create_behavioral_insights()
        pdf_insights = self.extract_key_pdf_insights()
        
        # Create focused, contextual structure
        optimized_data = {
            "survey_context": {
                "project": project_context["project_name"],
                "type": project_context["project_type"],
                "respondents": project_context["total_respondents"],
                "data_quality": f"Match confidence: {project_context['confidence_score']}/20",
                "summary": f"Survey data for {project_context['project_name']} with {project_context['total_respondents']} respondents"
            },
            
            "demographics": demographics,
            
            "satisfaction_ratings": satisfaction,
            
            "behavioral_data": behavioral,
            
            "key_insights": pdf_insights,
            
            "data_availability": {
                "demographic_questions": len(demographics),
                "satisfaction_metrics": len(satisfaction),
                "behavioral_insights": len(behavioral),
                "pdf_insights_available": len(pdf_insights) > 0
            },
            
            "response_patterns": self.create_response_patterns(),
            
            "gemini_context": {
                "optimized_for": "Google Gemini Flash 2.5",
                "structure": "Flattened and contextual",
                "focus": "Survey insights and statistical analysis",
                "data_processed": datetime.now().isoformat()
            }
        }
        
        return optimized_data
    
    def create_response_patterns(self) -> Dict[str, Any]:
        """Create response patterns for better Gemini understanding"""
        raw_sample = self.raw_data.get("responses_data", {}).get("raw_sample", [])
        
        if not raw_sample:
            return {"error": "No sample data available"}
        
        # Analyze sample responses
        sample_size = len(raw_sample)
        columns = list(raw_sample[0].keys()) if raw_sample else []
        
        patterns = {
            "sample_size": sample_size,
            "response_structure": f"{len(columns)} questions per respondent",
            "completion_patterns": self.analyze_completion_patterns(raw_sample),
            "example_response": raw_sample[0] if raw_sample else {}
        }
        
        return patterns
    
    def analyze_completion_patterns(self, sample_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how completely respondents filled out the survey"""
        if not sample_data:
            return {}
        
        completion_rates = []
        for response in sample_data:
            non_null_count = sum(1 for v in response.values() if v is not None and v != "")
            total_questions = len(response)
            completion_rate = (non_null_count / total_questions) * 100
            completion_rates.append(completion_rate)
        
        avg_completion = sum(completion_rates) / len(completion_rates)
        
        return {
            "average_completion_rate": f"{avg_completion:.1f}%",
            "range": f"{min(completion_rates):.1f}% - {max(completion_rates):.1f}%",
            "sample_analyzed": len(sample_data)
        }
    
    def create_gemini_prompt_template(self) -> Dict[str, str]:
        """Create ready-to-use prompt templates for Gemini"""
        project_name = self.raw_data.get("project_data", {}).get("current_project", {}).get("name", "Unknown")
        respondent_count = self.raw_data.get("responses_data", {}).get("file_info", {}).get("total_respondents", 0)
        
        templates = {
            "data_analysis": f"""Je bent een expert data analist. Je analyseert survey data voor {project_name} met {respondent_count} respondenten.

Gebruik alleen de data in de context. Vermeld altijd:
- Sample size ({respondent_count} respondenten)
- Percentages met context
- Databron wanneer relevant

Beschikbare data: demografische gegevens, tevredenheidscijfers, gedragsinzichten, en PDF rapport insights.""",
            
            "qa_system": f"""Je beantwoordt vragen over {project_name} survey data ({respondent_count} respondenten).

Regels:
- Gebruik alleen beschikbare data
- Vermeld altijd sample size
- Geef geen antwoord als data niet beschikbaar is
- Verwijs naar specifieke metrics wanneer mogelijk""",
            
            "report_generation": f"""Je genereert rapporten gebaseerd op {project_name} survey data.

Focus op:
- Belangrijkste bevindingen
- Statistisch significante resultaten
- Duidelijke conclusies gebaseerd op {respondent_count} respondenten
- Actionable insights voor stakeholders"""
        }
        
        return templates
    
    def save_optimized_data(self, optimized_data: Dict[str, Any], prompt_templates: Dict[str, str]) -> None:
        """Save all optimized files"""
        
        # Main optimized data file
        main_file = self.output_dir / "gemini_optimized_data.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Prompt templates
        templates_file = self.output_dir / "gemini_prompt_templates.json"
        with open(templates_file, 'w', encoding='utf-8') as f:
            json.dump(prompt_templates, f, indent=2, ensure_ascii=False, default=str)
        
        # Create ready-to-use context file
        context_file = self.output_dir / "gemini_context.json"
        ready_context = {
            "model_config": {
                "model": "gemini-1.5-flash",
                "temperature": 0.1,
                "max_output_tokens": 2048,
                "response_format": "text"
            },
            "data": optimized_data,
            "system_instructions": prompt_templates["data_analysis"],
            "usage_examples": [
                "Wat was de gemiddelde tevredenheid?",
                "Hoe was de demografische samenstelling?",
                "Wat waren de hoofdredenen voor bezoek?",
                "Welke inzichten komen uit het PDF rapport?"
            ]
        }
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(ready_context, f, indent=2, ensure_ascii=False, default=str)
        
        # Summary of optimization
        summary = {
            "optimization_summary": {
                "input_file": str(self.input_file),
                "output_files": [
                    "gemini_optimized_data.json",
                    "gemini_prompt_templates.json", 
                    "gemini_context.json"
                ],
                "optimizations_applied": [
                    "Flattened nested structures",
                    "Grouped by semantic meaning",
                    "Extracted key insights",
                    "Created response patterns",
                    "Generated prompt templates"
                ],
                "gemini_specific_features": [
                    "Contextual data organization",
                    "Ready-to-use prompt templates",
                    "Structured for Flash 2.5 capabilities",
                    "Optimized token usage"
                ],
                "data_reduction": {
                    "original_size": f"{self.input_file.stat().st_size / 1024:.1f} KB",
                    "optimized_focus": "Key insights and structured analysis"
                },
                "processing_timestamp": datetime.now().isoformat()
            }
        }
        
        summary_file = self.output_dir / "optimization_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Gemini optimization complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"Main data file: gemini_optimized_data.json")
        print(f"Prompt templates: gemini_prompt_templates.json")
        print(f"Ready context: gemini_context.json")
        print(f"Summary: optimization_summary.json")
    
    def optimize_for_gemini(self) -> None:
        """Main optimization process"""
        print("Optimizing survey data for Google Gemini Flash 2.5...")
        
        # Create optimized data structure
        optimized_data = self.create_gemini_optimized_data()
        
        # Create prompt templates
        prompt_templates = self.create_gemini_prompt_template()
        
        # Save all files
        self.save_optimized_data(optimized_data, prompt_templates)
        
        print("\nOptimization completed successfully!")
        print("Files are ready for use with Google Gemini Flash 2.5 API")


def main():
    """Main function"""
    if len(sys.argv) != 3:
        print("Usage: python gemini_optimizer.py <comprehensive_data_file> <output_directory>")
        print("Example: python gemini_optimizer.py './results/comprehensive_survey_data.json' './gemini_ready/'")
        return
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"ERROR: Input file '{input_file}' does not exist!")
        return
    
    try:
        optimizer = GeminiFlashDataOptimizer(input_file, output_dir)
        optimizer.optimize_for_gemini()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()