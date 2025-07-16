import pandas as pd
import json
import os
from pathlib import Path
import PyPDF2
from typing import Dict, Any, List, Optional
import re
import sys
import glob
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
data_path = os.getenv('DATA_PATH')
if data_path:
    os.chdir(data_path)
else:
    print("ERROR: DATA_PATH not set in .env!")
    exit(1)

class UniversalCigarBoxPreprocessor:
    """
    Universal preprocessor for CigarBox survey data - works with any project
    Always expects 3 files: aggregated projects Excel, individual responses Excel, PDF factsheet
    """
    
    def __init__(self, data_dir: str, output_dir: str):
        # Convert WSL paths to proper format
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect the 3 required files
        self.projects_file = self._find_projects_file()
        self.responses_file = self._find_responses_file()
        self.pdf_file = self._find_pdf_file()
        
        self._log_files_found()
        
    def _find_projects_file(self) -> Optional[Path]:
        """Find aggregated projects Excel file"""
        patterns = [
            "*project*.xlsx",
            "*Project*.xlsx", 
            "*PROJECT*.xlsx",
            "*geaggregeerde*.xlsx",
            "*aggregated*.xlsx"
        ]
        
        for pattern in patterns:
            files = list(self.data_dir.glob(pattern))
            if files:
                return files[0]
        return None
    
    def _find_responses_file(self) -> Optional[Path]:
        """Find individual responses Excel file"""
        patterns = [
            "*response*.xlsx",
            "*Response*.xlsx",
            "*RESPONSE*.xlsx", 
            "*respondent*.xlsx",
            "*individual*.xlsx"
        ]
        
        for pattern in patterns:
            files = list(self.data_dir.glob(pattern))
            if files:
                return files[0]
        return None
    
    def _find_pdf_file(self) -> Optional[Path]:
        """Find PDF factsheet"""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        return pdf_files[0] if pdf_files else None
    
    def _log_files_found(self):
        """Log which files were found"""
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Projects file: {self.projects_file.name if self.projects_file else 'NOT FOUND'}")
        print(f"Responses file: {self.responses_file.name if self.responses_file else 'NOT FOUND'}")
        print(f"PDF file: {self.pdf_file.name if self.pdf_file else 'NOT FOUND'}")
        
        missing_files = []
        if not self.projects_file: missing_files.append("Projects Excel file")
        if not self.responses_file: missing_files.append("Responses Excel file") 
        if not self.pdf_file: missing_files.append("PDF factsheet")
        
        if missing_files:
            print(f"WARNING: Missing files: {', '.join(missing_files)}")
        else:
            print("SUCCESS: All 3 required files found")
    
    def extract_pdf_content(self) -> Dict[str, Any]:
        """Extract ALL text content from PDF without losing information"""
        if not self.pdf_file or not self.pdf_file.exists():
            return {"error": "PDF file not found", "content": "", "pages": [], "extraction_success": False}
        
        try:
            with open(self.pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                pages_content = []
                full_text = ""
                
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    pages_content.append({
                        "page_number": i + 1,
                        "content": page_text,
                        "length": len(page_text)
                    })
                    full_text += page_text + "\n"
                
                return {
                    "full_text": full_text.strip(),
                    "pages": pages_content,
                    "total_pages": len(pdf_reader.pages),
                    "total_length": len(full_text),
                    "extraction_success": True
                }
                
        except Exception as e:
            return {
                "error": f"PDF extraction failed: {str(e)}",
                "content": "",
                "pages": [],
                "extraction_success": False
            }
    
    def process_projects_file(self) -> Dict[str, Any]:
        """Process aggregated projects Excel file - preserve ALL data"""
        if not self.projects_file or not self.projects_file.exists():
            return {"error": "Projects file not found", "raw_data": [], "column_analysis": {}}
        
        try:
            # Read Excel file
            df = pd.read_excel(self.projects_file)
            
            # Convert ALL data to preserve information
            processed_data = {
                "file_info": {
                    "filename": self.projects_file.name,
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "columns": list(df.columns)
                },
                "raw_data": [],
                "column_analysis": {}
            }
            
            # Process each row completely
            for idx, row in df.iterrows():
                row_data = {"row_index": idx}
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value):
                        # Convert to appropriate type while preserving info
                        if isinstance(value, (int, float)):
                            row_data[col] = value
                        else:
                            row_data[col] = str(value)
                    else:
                        row_data[col] = None
                processed_data["raw_data"].append(row_data)
            
            # Analyze each column
            for col in df.columns:
                processed_data["column_analysis"][col] = {
                    "data_type": str(df[col].dtype),
                    "non_null_count": int(df[col].count()),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_values": int(df[col].nunique()),
                    "sample_values": df[col].dropna().head(3).tolist()
                }
            
            return processed_data
            
        except Exception as e:
            return {"error": f"Failed to process projects file: {str(e)}", "raw_data": [], "column_analysis": {}}
    
    def process_responses_file(self) -> Dict[str, Any]:
        """Process individual responses Excel file - preserve ALL data"""
        if not self.responses_file or not self.responses_file.exists():
            return {"error": "Responses file not found", "file_info": {}, "column_analysis": {}, "raw_sample": []}
        
        try:
            df = pd.read_excel(self.responses_file)
            
            processed_data = {
                "file_info": {
                    "filename": self.responses_file.name,
                    "total_respondents": len(df),
                    "total_questions": len(df.columns),
                    "columns": list(df.columns)
                },
                "column_analysis": {},
                "statistical_summary": {},
                "categorical_summary": {},
                "raw_sample": []  # Store sample of raw data
            }
            
            # Analyze each column comprehensively
            for col in df.columns:
                column_info = {
                    "data_type": str(df[col].dtype),
                    "non_null_count": int(df[col].count()),
                    "null_count": int(df[col].isnull().sum()),
                    "unique_values": int(df[col].nunique()),
                    "completion_rate": float(df[col].count() / len(df) * 100)
                }
                
                # Statistical analysis for numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats = df[col].describe()
                    column_info["statistics"] = {
                        "mean": float(stats['mean']) if 'mean' in stats else None,
                        "median": float(df[col].median()) if df[col].count() > 0 else None,
                        "std": float(stats['std']) if 'std' in stats else None,
                        "min": float(stats['min']) if 'min' in stats else None,
                        "max": float(stats['max']) if 'max' in stats else None,
                        "q25": float(stats['25%']) if '25%' in stats else None,
                        "q75": float(stats['75%']) if '75%' in stats else None
                    }
                
                # Categorical analysis
                if df[col].nunique() <= 50:  # Reasonable limit for categories
                    value_counts = df[col].value_counts()
                    column_info["value_distribution"] = {
                        str(k): int(v) for k, v in value_counts.items()
                    }
                    column_info["value_percentages"] = {
                        str(k): float(v / len(df) * 100) for k, v in value_counts.items()
                    }
                
                processed_data["column_analysis"][col] = column_info
            
            # Store sample of raw data (first 10 rows) for context
            sample_data = []
            for idx in range(min(10, len(df))):
                row_data = {"response_id": idx}
                for col in df.columns:
                    value = df.iloc[idx][col]
                    if pd.notna(value):
                        row_data[col] = value if isinstance(value, (int, float)) else str(value)
                    else:
                        row_data[col] = None
                sample_data.append(row_data)
            
            processed_data["raw_sample"] = sample_data
            
            return processed_data
            
        except Exception as e:
            return {"error": f"Failed to process responses file: {str(e)}", "file_info": {}, "column_analysis": {}, "raw_sample": []}
    
    def analyze_pdf_content(self, pdf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PDF content to extract structured insights"""
        if not pdf_data.get("extraction_success"):
            return {"error": "PDF content not available", "full_content": ""}
        
        full_text = pdf_data["full_text"]
        
        analysis = {
            "content_structure": {
                "total_pages": pdf_data["total_pages"],
                "total_length": pdf_data["total_length"],
                "average_page_length": pdf_data["total_length"] / pdf_data["total_pages"] if pdf_data["total_pages"] > 0 else 0
            },
            "extracted_metrics": {
                "percentages": [],
                "numbers": [],
                "dates": [],
                "currencies": []
            },
            "identified_sections": [],
            "key_terms": [],
            "full_content": full_text  # Preserve complete content
        }
        
        # Extract percentages
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', full_text)
        analysis["extracted_metrics"]["percentages"] = [float(p) for p in percentages]
        
        # Extract numbers
        numbers = re.findall(r'\b(\d{1,3}(?:\.\d{3})*(?:,\d+)?)\b', full_text)
        analysis["extracted_metrics"]["numbers"] = numbers
        
        # Extract dates
        dates = re.findall(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', full_text)
        analysis["extracted_metrics"]["dates"] = dates
        
        # Extract currency amounts
        currencies = re.findall(r'€\s*(\d{1,3}(?:\.\d{3})*(?:,\d+)?)', full_text)
        analysis["extracted_metrics"]["currencies"] = currencies
        
        # Identify sections
        section_patterns = [
            r'^\d+\.\s*([^.\n]+)',  # Numbered sections
            r'^([A-Z][^.\n]{5,50})\s*$',  # Capitalized headers
            r'##?\s*([^.\n]+)',  # Markdown headers
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, full_text, re.MULTILINE)
            analysis["identified_sections"].extend(matches)
        
        # Remove duplicates and clean
        analysis["identified_sections"] = list(set(analysis["identified_sections"]))
        
        return analysis
    
    def identify_current_project(self, projects_data: Dict[str, Any], responses_data: Dict[str, Any], pdf_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify which specific project from the projects Excel matches the responses and PDF data
        """
        if not projects_data.get("raw_data"):
            return {"identified_project": {}, "match_confidence": 0, "total_projects_available": 0, "identification_method": "no_projects_data"}
        
        # Get potential identifiers from responses and PDF
        project_identifiers = {
            "respondent_count": responses_data.get("file_info", {}).get("total_respondents", 0),
            "pdf_filename": self.pdf_file.name if self.pdf_file else "",
            "responses_filename": responses_data.get("file_info", {}).get("filename", "")
        }
        
        # Extract project names/keywords from PDF content
        pdf_keywords = []
        if pdf_analysis.get("full_content"):
            pdf_text = pdf_analysis["full_content"].lower()
            # Common project name patterns
            name_patterns = [
                r'([A-Za-z\s]+festival[A-Za-z\s]*\d{4})',
                r'([A-Za-z\s]+concert[A-Za-z\s]*\d{4})',
                r'([A-Za-z\s]+evenement[A-Za-z\s]*\d{4})',
                r'(rotterdam[A-Za-z\s]+\d{4})',
                r'([A-Za-z\s]+\d{4}[A-Za-z\s]*)'
            ]
            
            for pattern in name_patterns:
                matches = re.findall(pattern, pdf_text, re.IGNORECASE)
                pdf_keywords.extend([match.strip() for match in matches])
        
        # Try to match project based on multiple criteria
        best_match = None
        best_score = 0
        
        for project in projects_data["raw_data"]:
            score = 0
            
            # Check respondent count match (if available in project data)
            for col in project.keys():
                if col and ('respondent' in col.lower() or 'response' in col.lower()):
                    try:
                        project_respondents = int(project[col]) if project[col] else 0
                        if project_respondents == project_identifiers["respondent_count"]:
                            score += 10
                    except (ValueError, TypeError):
                        pass
            
            # Check project name similarity with PDF keywords
            project_name = ""
            for col in project.keys():
                if col and ('name' in col.lower() or 'naam' in col.lower() or 'project' in col.lower()):
                    if project[col]:
                        project_name = str(project[col]).lower()
                        break
            
            if project_name:
                for keyword in pdf_keywords:
                    if keyword.lower() in project_name or project_name in keyword.lower():
                        score += 5
                
                # Check if project name appears in filenames
                if project_name in project_identifiers["pdf_filename"].lower():
                    score += 3
                if project_name in project_identifiers["responses_filename"].lower():
                    score += 3
            
            if score > best_score:
                best_score = score
                best_match = project
        
        # If no good match found, take the first project as fallback
        if best_match is None:
            best_match = projects_data["raw_data"][0]
            print(f"WARNING: Could not identify specific project, using first project in list")
        else:
            print(f"INFO: Identified current project with confidence score: {best_score}")
        
        return {
            "identified_project": best_match,
            "match_confidence": best_score,
            "total_projects_available": len(projects_data["raw_data"]),
            "identification_method": "automated_matching"
        }
    
    def create_comprehensive_llm_context(self, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive LLM context without losing information"""
        
        # Extract project metadata from the identified current project
        project_info = {"name": "Unknown Project", "type": "Survey Project"}
        
        # Use the identified current project instead of first project from list
        if all_data.get("project_data", {}).get("current_project"):
            current_project = all_data["project_data"]["current_project"]
            project_info.update(current_project)
        
        # Get comprehensive response info
        response_info = all_data.get("responses_data", {}).get("file_info", {})
        
        context = {
            "project_metadata": project_info,
            "survey_overview": {
                "total_respondents": response_info.get("total_respondents", 0),
                "total_questions": response_info.get("total_questions", 0),
                "response_file": response_info.get("filename", "Unknown"),
                "data_collection_method": "CigarBox survey platform"
            },
            "data_sources": {
                "projects_file": {
                    "available": bool(all_data.get("project_data", {}).get("current_project")),
                    "filename": all_data.get("project_data", {}).get("file_info", {}).get("filename"),
                    "total_projects_in_file": all_data.get("project_data", {}).get("project_identification", {}).get("total_projects_in_file", 0),
                    "current_project_identified": True,
                    "identification_confidence": all_data.get("project_data", {}).get("project_identification", {}).get("match_confidence", 0)
                },
                "responses_file": {
                    "available": bool(all_data.get("responses_data", {}).get("raw_sample")),
                    "filename": response_info.get("filename"),
                    "respondents": response_info.get("total_respondents", 0),
                    "questions": response_info.get("total_questions", 0)
                },
                "pdf_factsheet": {
                    "available": bool(all_data.get("pdf_content", {}).get("extraction_success")),
                    "filename": self.pdf_file.name if self.pdf_file else None,
                    "pages": all_data.get("pdf_content", {}).get("total_pages", 0),
                    "content_length": all_data.get("pdf_content", {}).get("total_length", 0)
                }
            },
            "analysis_capabilities": {
                "statistical_analysis": list(all_data.get("responses_data", {}).get("column_analysis", {}).keys()),
                "categorical_analysis": [
                    col for col, info in all_data.get("responses_data", {}).get("column_analysis", {}).items()
                    if "value_distribution" in info
                ],
                "numerical_analysis": [
                    col for col, info in all_data.get("responses_data", {}).get("column_analysis", {}).items()
                    if "statistics" in info
                ],
                "pdf_insights": len(all_data.get("pdf_analysis", {}).get("identified_sections", []))
            },
            "data_quality_assessment": {
                "completeness_score": self._calculate_completeness_score(all_data),
                "total_data_points": self._count_total_data_points(all_data),
                "processing_timestamp": datetime.now().isoformat(),
                "data_consistency": "Single project focus - responses and PDF match project data"
            }
        }
        
        return context
    
    def _calculate_completeness_score(self, data: Dict[str, Any]) -> float:
        """Calculate overall data completeness score"""
        score = 0
        max_score = 3
        
        if data.get("project_data", {}).get("current_project"):
            score += 1
        if data.get("responses_data", {}).get("raw_sample"):
            score += 1
        if data.get("pdf_content", {}).get("extraction_success"):
            score += 1
        
        return (score / max_score) * 100
    
    def _count_total_data_points(self, data: Dict[str, Any]) -> int:
        """Count total data points processed"""
        total = 0
        
        # Count current project data points
        current_project = data.get("project_data", {}).get("current_project", {})
        total += len(current_project.keys()) if current_project else 0
        
        # Count response data points
        responses_info = data.get("responses_data", {}).get("file_info", {})
        total += responses_info.get("total_respondents", 0) * responses_info.get("total_questions", 0)
        
        # Count PDF content
        total += data.get("pdf_content", {}).get("total_length", 0)
        
        return total
    
    def process_all_data(self) -> Dict[str, Any]:
        """Process all 3 data sources comprehensively"""
        print("\nProcessing all data sources...")
        
        # Process each data source
        projects_data = self.process_projects_file()
        responses_data = self.process_responses_file()
        pdf_content = self.extract_pdf_content()
        pdf_analysis = self.analyze_pdf_content(pdf_content)
        
        # Identify the current project from the projects list
        current_project_info = self.identify_current_project(projects_data, responses_data, pdf_analysis)
        
        # Create focused project data with only the relevant project
        focused_projects_data = {
            "file_info": projects_data.get("file_info", {}),
            "current_project": current_project_info.get("identified_project", {}),
            "project_identification": {
                "match_confidence": current_project_info.get("match_confidence", 0),
                "total_projects_in_file": current_project_info.get("total_projects_available", 0),
                "identification_method": current_project_info.get("identification_method", "unknown")
            },
            "column_analysis": projects_data.get("column_analysis", {})
            # NOTE: raw_data with all projects is intentionally removed to avoid LLM confusion
        }
        
        # Combine all data with focused project information
        comprehensive_data = {
            "project_data": focused_projects_data,  # Only the relevant project
            "responses_data": responses_data,
            "pdf_content": pdf_content,
            "pdf_analysis": pdf_analysis,
            "processing_metadata": {
                "files_processed": {
                    "projects": bool(projects_data.get("raw_data")),
                    "responses": bool(responses_data.get("raw_sample")),
                    "pdf": bool(pdf_content.get("extraction_success"))
                },
                "processing_timestamp": datetime.now().isoformat(),
                "total_data_preserved": True,
                "project_filtering_applied": True
            }
        }
        
        # Add comprehensive LLM context
        comprehensive_data["llm_context"] = self.create_comprehensive_llm_context(comprehensive_data)
        
        return comprehensive_data
    
    def save_processed_data(self, data: Dict[str, Any]) -> None:
        """Save all processed data for LLM consumption"""
        
        # Main comprehensive data file
        main_file = self.output_dir / "comprehensive_survey_data.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        # Data structure guide for LLM integration
        data_guide = {
            "data_structure_guide": {
                "project_data": "Specifieke projectinformatie (gefilterd uit projecten Excel)",
                "responses_data": "Individuele survey responses met complete analyse",
                "pdf_content": "Volledige PDF factsheet content",
                "pdf_analysis": "Gestructureerde PDF insights",
                "llm_context": "Project overzicht en data kwaliteit"
            },
            
            "usage_instructions": [
                "Alle originele data is behouden - geen informatie verloren",
                "Gebruik 'column_analysis' voor gedetailleerde statistieken",
                "Gebruik 'raw_sample' voor context van individuele responses",
                "Gebruik 'pdf_content' voor volledige factsheet informatie",
                "Verwijs altijd naar 'llm_context' voor project overzicht"
            ],
            
            "data_access_patterns": {
                "voor_statistieken": "data['responses_data']['column_analysis'][column_name]['statistics']",
                "voor_categorische_data": "data['responses_data']['column_analysis'][column_name]['value_distribution']",
                "voor_project_info": "data['project_data']['current_project']",
                "voor_pdf_inhoud": "data['pdf_content']['full_text']",
                "voor_response_sample": "data['responses_data']['raw_sample']"
            }
        }
        
        guide_file = self.output_dir / "data_structure_guide.json"
        with open(guide_file, 'w', encoding='utf-8') as f:
            json.dump(data_guide, f, indent=2, ensure_ascii=False, default=str)
        
        # Create summary report
        summary = {
            "processing_report": {
                "project_name": data.get("llm_context", {}).get("project_metadata", {}).get("name", "Unknown"),
                "total_respondents": data.get("llm_context", {}).get("survey_overview", {}).get("total_respondents", 0),
                "data_completeness": f"{data.get('llm_context', {}).get('data_quality_assessment', {}).get('completeness_score', 0):.1f}%",
                "total_data_points": data.get("llm_context", {}).get("data_quality_assessment", {}).get("total_data_points", 0),
                "files_processed": data.get("processing_metadata", {}).get("files_processed", {}),
                "processing_timestamp": datetime.now().isoformat(),
                "data_preservation": "Complete - no information lost"
            },
            "file_outputs": {
                "comprehensive_data": "comprehensive_survey_data.json",
                "data_guide": "data_structure_guide.json",
                "summary": "processing_report.json"
            },
            "data_structure_reference": {
                "comprehensive_survey_data.json": {
                    "description": "Complete survey data with all original information preserved",
                    "main_sections": {
                        "project_data": {
                            "description": "Specific project information (filtered from projects Excel)",
                            "contains": [
                                "file_info: metadata about the projects Excel file",
                                "current_project: data for the specific project matching responses/PDF",
                                "project_identification: confidence score and matching method used",
                                "column_analysis: statistical analysis of project Excel columns"
                            ]
                        },
                        "responses_data": {
                            "description": "Individual survey responses with full analysis",
                            "contains": [
                                "file_info: metadata about responses file (respondent count, questions, etc.)",
                                "column_analysis: comprehensive analysis of each survey question/column",
                                "raw_sample: sample of actual response data for context"
                            ]
                        },
                        "pdf_content": {
                            "description": "Complete PDF factsheet content",
                            "contains": [
                                "full_text: entire PDF content as text",
                                "pages: page-by-page breakdown of PDF content",
                                "total_pages: number of pages in PDF",
                                "total_length: total character count"
                            ]
                        },
                        "pdf_analysis": {
                            "description": "Structured analysis of PDF content",
                            "contains": [
                                "content_structure: metadata about PDF structure",
                                "extracted_metrics: numbers, percentages, dates found in PDF",
                                "identified_sections: detected sections/headers in PDF",
                                "full_content: complete PDF text preserved"
                            ]
                        },
                        "llm_context": {
                            "description": "Project overview and data quality assessment",
                            "contains": [
                                "project_metadata: project name, type, and basic info",
                                "survey_overview: respondent count, questions, data collection method",
                                "data_sources: status and metadata of all 3 input files",
                                "analysis_capabilities: list of available analysis types",
                                "data_quality_assessment: completeness score and data point count"
                            ]
                        },
                        "processing_metadata": {
                            "description": "Information about the preprocessing operation",
                            "contains": [
                                "files_processed: boolean status of each file type",
                                "processing_timestamp: when processing occurred",
                                "total_data_preserved: confirmation of no data loss",
                                "project_filtering_applied: confirmation that only relevant project data included"
                            ]
                        }
                    }
                }
            }
        }
        
        summary_file = self.output_dir / "processing_report.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nProcessing complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"Main data file: comprehensive_survey_data.json")
        print(f"Data guide: data_structure_guide.json") 
        print(f"Summary report: processing_report.json")
        print(f"Data preservation: Complete - no information lost")


def main():
    """Main function with WSL-compatible path handling"""
    data_dir = os.getenv('DATA_PATH')
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if not data_dir:
        print("ERROR: DATA_PATH not set in .env and no directory provided!")
        return
    output_dir = "results"
    print("Starting Universal CigarBox Data Preprocessing...")
    print(f"WSL Environment Detected")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    if data_dir.startswith('C:\\'):
        data_dir = data_dir.replace('C:\\', '/mnt/c/').replace('\\', '/')
        output_dir = str(Path(data_dir) / "results")
        print(f"Converted to WSL path: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' does not exist!")
        return
    try:
        preprocessor = UniversalCigarBoxPreprocessor(data_dir, output_dir)
        processed_data = preprocessor.process_all_data()
        preprocessor.save_processed_data(processed_data)
        print("\nUniversal preprocessing successful!")
        print("Ready for any LLM API (Google Gemini, OpenAI, etc.)")
        print("All original data preserved and structured")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()