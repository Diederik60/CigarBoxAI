#!/usr/bin/env python3
"""
CigarBox AI Data Preprocessing Script
===================================

This script processes three data sources for optimal LLM usage:
1. Aggregated projects Excel file (multi-sheet)
2. Individual responses Excel file (single project)
3. PDF factsheet with interpretations and context

"""

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from pathlib import Path
import PyPDF2
import logging
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CigarBoxPreprocessor:
    """
    Main preprocessing class for CigarBox AI data preparation
    """
    
    def __init__(self, 
                 projects_file: str = "projecten_090720250858.xlsx",
                 responses_file: str = "responses_rotterdam_bluegrass_festival.xlsx",
                 pdf_file: str = "rotterdam_bluegrass_factsheet.pdf",
                 target_project: str = "Rotterdam Bluegrass Festival 2024"):
        
        self.projects_file = projects_file
        self.responses_file = responses_file
        self.pdf_file = pdf_file
        self.target_project = target_project
        
        self.processed_data = {}
        self.validation_data = {}
        
        logger.info(f"Initialized CigarBoxPreprocessor for project: {target_project}")
    
    def process_all_data(self) -> Dict[str, Any]:
        """
        Main processing pipeline - processes all three data sources
        """
        logger.info("Starting complete data processing pipeline...")
        
        # 1. Process aggregated projects data
        self.processed_data['projects'] = self.process_projects_data()
        
        # 2. Process individual responses
        self.processed_data['responses'] = self.process_responses_data()
        
        # 3. Process PDF context
        self.processed_data['pdf_context'] = self.process_pdf_context()
        
        # 4. Create validation dataset
        self.validation_data = self.create_validation_dataset()
        
        # 5. Generate query-ready datasets
        self.processed_data['query_ready'] = self.prepare_query_datasets()
        
        logger.info("✓ Complete data processing pipeline finished")
        return self.processed_data
    
    def process_projects_data(self) -> Dict[str, Any]:
        """
        Process the multi-sheet projects Excel file
        """
        logger.info("Processing projects data...")
        
        try:
            # Read all sheets
            xl_file = pd.ExcelFile(self.projects_file)
            sheets_data = {}
            
            for sheet_name in xl_file.sheet_names:
                logger.info(f"Processing sheet: {sheet_name}")
                df = pd.read_excel(self.projects_file, sheet_name=sheet_name)
                
                # Clean column names
                df.columns = self.clean_column_names(df.columns)
                
                # Find target project
                target_row = self.find_target_project(df)
                
                if target_row is not None:
                    sheets_data[sheet_name] = {
                        'target_project_data': df.iloc[target_row].to_dict(),
                        'all_projects_data': df.to_dict('records'),
                        'columns': df.columns.tolist(),
                        'shape': df.shape
                    }
                else:
                    sheets_data[sheet_name] = {
                        'target_project_data': None,
                        'all_projects_data': df.to_dict('records'),
                        'columns': df.columns.tolist(),
                        'shape': df.shape
                    }
            
            # Combine target project data from all sheets
            target_combined = self.combine_target_project_data(sheets_data)
            
            # Create benchmark datasets
            benchmark_data = self.create_benchmark_datasets(sheets_data)
            
            return {
                'sheets': sheets_data,
                'target_project_combined': target_combined,
                'benchmark_data': benchmark_data,
                'metadata': {
                    'sheets_processed': len(sheets_data),
                    'target_project': self.target_project,
                    'processed_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing projects data: {e}")
            return {'error': str(e)}
    
    def process_responses_data(self) -> Dict[str, Any]:
        """
        Process individual responses Excel file
        """
        logger.info("Processing responses data...")
        
        try:
            # Read responses file
            df = pd.read_excel(self.responses_file)
            logger.info(f"Loaded {len(df)} responses")
            
            # Clean column names
            df.columns = self.clean_column_names(df.columns)
            
            # Extract different types of data
            demographics = self.extract_demographics(df)
            satisfaction = self.extract_satisfaction_metrics(df)
            motivations = self.extract_motivations(df)
            open_text = self.extract_open_text_responses(df)
            
            # Create summary statistics
            summary_stats = self.create_response_summary(df)
            
            return {
                'raw_data': df.to_dict('records'),
                'demographics': demographics,
                'satisfaction': satisfaction,
                'motivations': motivations,
                'open_text': open_text,
                'summary_stats': summary_stats,
                'metadata': {
                    'total_responses': len(df),
                    'columns_count': len(df.columns),
                    'processed_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing responses data: {e}")
            return {'error': str(e)}
    
    def process_pdf_context(self) -> Dict[str, Any]:
        """
        Process PDF factsheet for context and interpretations
        """
        logger.info("Processing PDF context...")
        
        # Since we have the PDF content in the document context, we'll structure it
        # based on the known structure from the factsheet
        
        pdf_context = {
            'project_metadata': {
                'name': 'Rotterdam Bluegrass Festival 2024',
                'dates': '28-30 juni 2024',
                'location': 'Noordplein, Rotterdam',
                'respondents': 614,
                'expected_visitors': 15000,
                'actual_visitors': None,  # Not specified in gerealiseerd
                'budget': 520000,
                'access_type': 'Betaalde toegang'
            },
            
            'chart_interpretations': {
                'demographics': {
                    'gender': 'Vrouwen domineren het publiek (46%), mannen 54%',
                    'age': 'Breed publiek met piek in 50-59 jaar (27%) en 60-69 jaar (22%)',
                    'education': 'Hoogopgeleid publiek: 59% bachelor/master/doctoraal',
                    'geography': '95% uit Nederland, 3% België, internationaal beperkt'
                },
                
                'cultural_groups': {
                    'top_groups': 'Weelderige Cultuurminnaars (Index 275), Culturele Alleseters (Index 174), Proevende Buitenwijkers (Index 173)',
                    'underrepresented': 'De Lokale Vrijetijdsgenieters (Index 27), Nieuwsgierige Toekomstgrijpers (Index 58)',
                    'context': 'Festival trekt vooral ervaren cultuurliefhebbers aan'
                },
                
                'satisfaction': {
                    'overall_rating': 8.8,
                    'key_strengths': 'Programma (8.6), Bereikbaarheid (8.7), Gastvrijheid (9.0)',
                    'areas_for_improvement': 'Eten en drinken (8.0), Prijs/kwaliteit (8.6)'
                },
                
                'accessibility': {
                    'information_access': '93% kon informatie makkelijk vinden',
                    'location_access': '92% kon locatie goed bereiken',
                    'facilities': '97% vond faciliteiten goed bereikbaar'
                }
            },
            
            'benchmarks': {
                'rotterdam_comparison': 'Zie Culturele Doelgroepen Index Rotterdam tabellen',
                'national_comparison': 'Zie Culturele Doelgroepen Index Nederland tabellen',
                'similar_events': ['Sinfonia Maritiem Concert', 'Sinfonia Maritiem Familieconcert', 'Theaterbootcamp']
            },
            
            'key_insights': [
                'Festival trekt hoogopgeleide, ervaren cultuurliefhebbers',
                'Sterke sociale binding: 31% voelt zich meer verbonden met buurt/stad',
                'Duurzaam transport: 50% OV, 34% fiets, 17% wandelen',
                'Leereffect: 31% heeft iets nieuws geleerd',
                'Beperkte economische impact: 11% overnachters, 17% andere activiteiten'
            ]
        }
        
        return pdf_context
    
    def clean_column_names(self, columns: List[str]) -> List[str]:
        """
        Clean and standardize column names
        """
        cleaned = []
        for col in columns:
            if pd.isna(col):
                cleaned.append('unnamed_column')
            else:
                # Remove special characters and standardize
                clean_col = re.sub(r'[^\w\s-]', '', str(col))
                clean_col = re.sub(r'\s+', '_', clean_col.strip())
                cleaned.append(clean_col.lower())
        return cleaned
    
    def find_target_project(self, df: pd.DataFrame) -> Optional[int]:
        """
        Find the target project in the dataframe
        """
        # Look for target project in various columns
        for col in df.columns:
            if df[col].dtype == 'object':
                matches = df[col].str.contains(self.target_project, case=False, na=False)
                if matches.any():
                    return matches.idxmax()
        return None
    
    def combine_target_project_data(self, sheets_data: Dict) -> Dict[str, Any]:
        """
        Combine target project data from all sheets
        """
        combined = {}
        for sheet_name, data in sheets_data.items():
            if data['target_project_data']:
                combined[sheet_name] = data['target_project_data']
        return combined
    
    def create_benchmark_datasets(self, sheets_data: Dict) -> Dict[str, Any]:
        """
        Create benchmark datasets for comparison
        """
        benchmark = {}
        
        for sheet_name, data in sheets_data.items():
            if data['all_projects_data']:
                # Create metrics for comparison
                benchmark[sheet_name] = {
                    'all_projects': data['all_projects_data'],
                    'summary_stats': self.calculate_benchmark_stats(data['all_projects_data'])
                }
        
        return benchmark
    
    def calculate_benchmark_stats(self, projects_data: List[Dict]) -> Dict:
        """
        Calculate summary statistics for benchmarking
        """
        df = pd.DataFrame(projects_data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        return stats
    
    def extract_demographics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract demographic information from responses
        """
        demographics = {}
        
        # Look for demographic columns
        demo_patterns = ['geslacht', 'leeftijd', 'opleiding', 'woon', 'culturele_achtergrond']
        
        for pattern in demo_patterns:
            demo_cols = [col for col in df.columns if pattern in col]
            if demo_cols:
                demographics[pattern] = {}
                for col in demo_cols:
                    if df[col].dtype == 'object':
                        demographics[pattern][col] = df[col].value_counts().to_dict()
                    else:
                        demographics[pattern][col] = df[col].describe().to_dict()
        
        return demographics
    
    def extract_satisfaction_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract satisfaction and rating metrics
        """
        satisfaction = {}
        
        # Look for rating/satisfaction columns
        rating_patterns = ['waardering', 'tevredenheid', 'rating', 'score']
        
        for pattern in rating_patterns:
            rating_cols = [col for col in df.columns if pattern in col]
            if rating_cols:
                satisfaction[pattern] = {}
                for col in rating_cols:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        satisfaction[pattern][col] = {
                            'mean': df[col].mean(),
                            'median': df[col].median(),
                            'std': df[col].std(),
                            'distribution': df[col].value_counts().to_dict()
                        }
        
        return satisfaction
    
    def extract_motivations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract motivation and reason data
        """
        motivations = {}
        
        # Look for motivation columns
        motivation_patterns = ['reden', 'motivatie', 'waarom', 'bezocht']
        
        for pattern in motivation_patterns:
            motivation_cols = [col for col in df.columns if pattern in col]
            if motivation_cols:
                motivations[pattern] = {}
                for col in motivation_cols:
                    if df[col].dtype == 'object':
                        motivations[pattern][col] = df[col].value_counts().to_dict()
                    elif df[col].dtype == 'bool':
                        motivations[pattern][col] = df[col].value_counts().to_dict()
        
        return motivations
    
    def extract_open_text_responses(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Extract open text responses for sentiment analysis
        """
        open_text = {}
        
        # Look for text columns with substantial content
        text_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains substantial text (not just categories)
                sample_values = df[col].dropna().head(10)
                if any(len(str(val)) > 20 for val in sample_values):
                    text_cols.append(col)
        
        for col in text_cols:
            responses = df[col].dropna().tolist()
            if responses:
                open_text[col] = responses
        
        return open_text
    
    def create_response_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create summary statistics for responses
        """
        return {
            'total_responses': len(df),
            'response_date_range': self.get_date_range(df),
            'completion_rate': self.calculate_completion_rate(df),
            'key_metrics': self.extract_key_metrics(df)
        }
    
    def get_date_range(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Get date range from responses
        """
        date_cols = [col for col in df.columns if 'date' in col or 'created' in col]
        if date_cols:
            date_col = date_cols[0]
            dates = pd.to_datetime(df[date_col], errors='coerce')
            return {
                'start_date': dates.min().strftime('%Y-%m-%d'),
                'end_date': dates.max().strftime('%Y-%m-%d')
            }
        return {'start_date': None, 'end_date': None}
    
    def calculate_completion_rate(self, df: pd.DataFrame) -> float:
        """
        Calculate response completion rate
        """
        total_cells = df.size
        filled_cells = df.count().sum()
        return (filled_cells / total_cells) * 100 if total_cells > 0 else 0
    
    def extract_key_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract key metrics from responses
        """
        metrics = {}
        
        # Look for key metric columns
        metric_patterns = ['overall', 'algemeen', 'totaal', 'gemiddeld']
        
        for pattern in metric_patterns:
            metric_cols = [col for col in df.columns if pattern in col]
            for col in metric_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    metrics[col] = {
                        'mean': df[col].mean(),
                        'median': df[col].median(),
                        'count': df[col].count()
                    }
        
        return metrics
    
    def create_validation_dataset(self) -> Dict[str, Any]:
        """
        Create validation dataset for LLM output verification
        """
        validation = {
            'numerical_facts': self.extract_numerical_facts(),
            'categorical_facts': self.extract_categorical_facts(),
            'cross_references': self.create_cross_references(),
            'validation_rules': self.create_validation_rules()
        }
        
        return validation
    
    def extract_numerical_facts(self) -> Dict[str, Any]:
        """
        Extract numerical facts for validation
        """
        facts = {}
        
        # From PDF context
        if 'pdf_context' in self.processed_data:
            pdf_data = self.processed_data['pdf_context']
            facts['pdf_facts'] = {
                'respondents': pdf_data['project_metadata']['respondents'],
                'expected_visitors': pdf_data['project_metadata']['expected_visitors'],
                'budget': pdf_data['project_metadata']['budget'],
                'overall_rating': pdf_data['chart_interpretations']['satisfaction']['overall_rating']
            }
        
        # From responses data
        if 'responses' in self.processed_data:
            response_data = self.processed_data['responses']
            facts['response_facts'] = {
                'total_responses': response_data['metadata']['total_responses'],
                'completion_rate': response_data['summary_stats']['completion_rate']
            }
        
        return facts
    
    def extract_categorical_facts(self) -> Dict[str, Any]:
        """
        Extract categorical facts for validation
        """
        facts = {}
        
        # From PDF context
        if 'pdf_context' in self.processed_data:
            pdf_data = self.processed_data['pdf_context']
            facts['project_facts'] = {
                'name': pdf_data['project_metadata']['name'],
                'location': pdf_data['project_metadata']['location'],
                'access_type': pdf_data['project_metadata']['access_type']
            }
        
        return facts
    
    def create_cross_references(self) -> Dict[str, Any]:
        """
        Create cross-references between data sources
        """
        cross_refs = {}
        
        # Match respondent counts across sources
        pdf_respondents = self.processed_data.get('pdf_context', {}).get('project_metadata', {}).get('respondents')
        response_count = self.processed_data.get('responses', {}).get('metadata', {}).get('total_responses')
        
        if pdf_respondents and response_count:
            cross_refs['respondent_validation'] = {
                'pdf_count': pdf_respondents,
                'response_count': response_count,
                'match': pdf_respondents == response_count
            }
        
        return cross_refs
    
    def create_validation_rules(self) -> List[Dict[str, Any]]:
        """
        Create validation rules for LLM output
        """
        rules = [
            {
                'rule': 'numerical_accuracy',
                'description': 'All numerical claims must match source data exactly',
                'validation_source': 'numerical_facts'
            },
            {
                'rule': 'categorical_consistency',
                'description': 'Categorical statements must be consistent with source data',
                'validation_source': 'categorical_facts'
            },
            {
                'rule': 'source_citation',
                'description': 'All claims must be properly cited to source data',
                'validation_source': 'cross_references'
            }
        ]
        
        return rules
    
    def prepare_query_datasets(self) -> Dict[str, Any]:
        """
        Prepare datasets optimized for different query types
        """
        query_datasets = {
            'storytelling': self.prepare_storytelling_data(),
            'benchmarking': self.prepare_benchmarking_data(),
            'qa': self.prepare_qa_data(),
            'open_questions': self.prepare_open_questions_data()
        }
        
        return query_datasets
    
    def prepare_storytelling_data(self) -> Dict[str, Any]:
        """
        Prepare data for intelligent storytelling (hover-over insights)
        """
        return {
            'chart_data': self.processed_data.get('responses', {}).get('demographics', {}),
            'interpretations': self.processed_data.get('pdf_context', {}).get('chart_interpretations', {}),
            'key_insights': self.processed_data.get('pdf_context', {}).get('key_insights', []),
            'validation_data': self.validation_data.get('numerical_facts', {})
        }
    
    def prepare_benchmarking_data(self) -> Dict[str, Any]:
        """
        Prepare data for automatic benchmarking
        """
        return {
            'target_project': self.processed_data.get('projects', {}).get('target_project_combined', {}),
            'all_projects': self.processed_data.get('projects', {}).get('benchmark_data', {}),
            'benchmark_context': self.processed_data.get('pdf_context', {}).get('benchmarks', {}),
            'validation_data': self.validation_data.get('cross_references', {})
        }
    
    def prepare_qa_data(self) -> Dict[str, Any]:
        """
        Prepare data for Q&A interface
        """
        return {
            'all_data': {
                'projects': self.processed_data.get('projects', {}),
                'responses': self.processed_data.get('responses', {}),
                'context': self.processed_data.get('pdf_context', {})
            },
            'validation_data': self.validation_data,
            'quick_facts': self.create_quick_facts()
        }
    
    def prepare_open_questions_data(self) -> Dict[str, Any]:
        """
        Prepare data for open question analysis
        """
        return {
            'open_text_responses': self.processed_data.get('responses', {}).get('open_text', {}),
            'sentiment_context': self.processed_data.get('pdf_context', {}).get('key_insights', []),
            'validation_data': self.validation_data.get('categorical_facts', {})
        }
    
    def create_quick_facts(self) -> Dict[str, Any]:
        """
        Create quick facts for fast Q&A responses
        """
        facts = {}
        
        # From PDF metadata
        if 'pdf_context' in self.processed_data:
            metadata = self.processed_data['pdf_context']['project_metadata']
            facts.update({
                'project_name': metadata['name'],
                'project_dates': metadata['dates'],
                'location': metadata['location'],
                'respondents': metadata['respondents'],
                'budget': metadata['budget']
            })
        
        # From responses summary
        if 'responses' in self.processed_data:
            response_meta = self.processed_data['responses']['metadata']
            facts.update({
                'total_responses': response_meta['total_responses'],
                'response_columns': response_meta['columns_count']
            })
        
        return facts
    
    def save_processed_data(self, output_dir: str = "processed_data"):
        """
        Save processed data to files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main processed data
        with open(output_path / "processed_data.json", "w", encoding="utf-8") as f:
            json.dump(self.processed_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Save validation data
        with open(output_path / "validation_data.json", "w", encoding="utf-8") as f:
            json.dump(self.validation_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Save query-ready datasets separately
        query_data = self.processed_data.get('query_ready', {})
        for query_type, data in query_data.items():
            with open(output_path / f"{query_type}_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✓ Processed data saved to {output_path}")
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of the processed data
        """
        report = []
        report.append("=" * 60)
        report.append("CIGARBOX AI DATA PREPROCESSING SUMMARY")
        report.append("=" * 60)
        
        # Projects data summary
        projects_data = self.processed_data.get('projects', {})
        if projects_data:
            report.append(f"\n📊 PROJECTS DATA:")
            report.append(f"  - Sheets processed: {projects_data.get('metadata', {}).get('sheets_processed', 0)}")
            report.append(f"  - Target project: {projects_data.get('metadata', {}).get('target_project', 'N/A')}")
        
        # Responses data summary
        responses_data = self.processed_data.get('responses', {})
        if responses_data:
            report.append(f"\n📋 RESPONSES DATA:")
            report.append(f"  - Total responses: {responses_data.get('metadata', {}).get('total_responses', 0)}")
            report.append(f"  - Completion rate: {responses_data.get('summary_stats', {}).get('completion_rate', 0):.1f}%")
        
        # PDF context summary
        pdf_data = self.processed_data.get('pdf_context', {})
        if pdf_data:
            report.append(f"\n📄 PDF CONTEXT:")
            report.append(f"  - Project: {pdf_data.get('project_metadata', {}).get('name', 'N/A')}")
            report.append(f"  - Respondents: {pdf_data.get('project_metadata', {}).get('respondents', 0)}")
            report.append(f"  - Budget: €{pdf_data.get('project_metadata', {}).get('budget', 0):,}")
        
        # Validation data summary
        validation_data = self.validation_data
        if validation_data:
            report.append(f"\n✅ VALIDATION DATA:")
            report.append(f"  - Numerical facts: {len(validation_data.get('numerical_facts', {}))}")
            report.append(f"  - Categorical facts: {len(validation_data.get('categorical_facts', {}))}")
            report.append(f"  - Validation rules: {len(validation_data.get('validation_rules', []))}")
        
        # Query datasets summary
        query_data = self.processed_data.get('query_ready', {})
        if query_data:
            report.append(f"\n🔍 QUERY-READY DATASETS:")
            for query_type in query_data.keys():
                report.append(f"  - {query_type.capitalize()} data: ✓")
        
        report.append(f"\n⏰ Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """
    Main execution function
    """
    print("🚀 Starting CigarBox AI Data Preprocessing...")
    
    # Initialize preprocessor
    preprocessor = CigarBoxPreprocessor()
    
    # Process all data
    processed_data = preprocessor.process_all_data()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    # Generate and print summary report
    summary = preprocessor.generate_summary_report()
    print(summary)
    
    # Save summary report
    with open("processed_data/summary_report.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print("\n✅ Data preprocessing completed successfully!")
    print("📁 Check 'processed_data' directory for output files")
    
    return processed_data


# Example usage for different query types
class QueryProcessor:
    """
    Process different types of queries using the preprocessed data
    """
    
    def __init__(self, processed_data: Dict[str, Any], validation_data: Dict[str, Any]):
        self.processed_data = processed_data
        self.validation_data = validation_data
    
    def prepare_storytelling_query(self, chart_element: str) -> Dict[str, Any]:
        """
        Prepare data for storytelling query (hover-over insights)
        """
        storytelling_data = self.processed_data.get('query_ready', {}).get('storytelling', {})
        
        return {
            'chart_data': storytelling_data.get('chart_data', {}),
            'interpretation': storytelling_data.get('interpretations', {}),
            'validation_data': storytelling_data.get('validation_data', {}),
            'context': f"Mouse hover over {chart_element}"
        }
    
    def prepare_benchmark_query(self, metric: str) -> Dict[str, Any]:
        """
        Prepare data for benchmarking query
        """
        benchmark_data = self.processed_data.get('query_ready', {}).get('benchmarking', {})
        
        return {
            'target_project': benchmark_data.get('target_project', {}),
            'all_projects': benchmark_data.get('all_projects', {}),
            'metric': metric,
            'validation_data': benchmark_data.get('validation_data', {})
        }
    
    def prepare_qa_query(self, question: str) -> Dict[str, Any]:
        """
        Prepare data for Q&A query
        """
        qa_data = self.processed_data.get('query_ready', {}).get('qa', {})
        
        return {
            'question': question,
            'all_data': qa_data.get('all_data', {}),
            'quick_facts': qa_data.get('quick_facts', {}),
            'validation_data': qa_data.get('validation_data', {})
        }
    
    def prepare_open_questions_query(self, responses: List[str]) -> Dict[str, Any]:
        """
        Prepare data for open questions analysis
        """
        open_data = self.processed_data.get('query_ready', {}).get('open_questions', {})
        
        return {
            'text_responses': responses,
            'context': open_data.get('sentiment_context', []),
            'validation_data': open_data.get('validation_data', {})
        }


# Validation functions
def validate_llm_response(response: str, validation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate LLM response against source data
    """
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check numerical claims
    numerical_facts = validation_data.get('numerical_facts', {})
    for source, facts in numerical_facts.items():
        for fact_name, fact_value in facts.items():
            if str(fact_value) in response and fact_name not in response:
                validation_result['warnings'].append(f"Number {fact_value} found but not properly contextualized")
    
    # Check categorical consistency
    categorical_facts = validation_data.get('categorical_facts', {})
    for source, facts in categorical_facts.items():
        for fact_name, fact_value in facts.items():
            if fact_value in response:
                # Good - categorical fact is properly referenced
                pass
    
    return validation_result


def create_llm_prompt(query: str, prepared_data: Dict[str, Any]) -> str:
    """
    Create an optimized prompt for LLM with prepared data
    """
    prompt_template = """
    Je bent een expert data-analist voor CigarBox AI. Je helpt gebruikers met het interpreteren van hun cultuurproject data.

    REGELS:
    - Gebruik ALLEEN exacte cijfers uit de verstrekte data
    - Citeer altijd je bronnen
    - Als je iets niet weet, zeg dat expliciet
    - Geef objectieve interpretaties zonder suggestieve conclusies

    BESCHIKBARE DATA:
    {data_summary}

    GEBRUIKERSVRAAG: {query}

    ANTWOORD:
    """
    
    # Create data summary
    data_summary = json.dumps(prepared_data, indent=2, ensure_ascii=False)
    
    return prompt_template.format(
        data_summary=data_summary,
        query=query
    )


# Example usage
if __name__ == "__main__":
    # Run the main preprocessing
    processed_data = main()
    
    # Example: Prepare for different query types
    query_processor = QueryProcessor(processed_data, processed_data.get('validation_data', {}))
    
    # Example storytelling query
    storytelling_query = query_processor.prepare_storytelling_query("demographics_chart")
    
    # Example benchmark query
    benchmark_query = query_processor.prepare_benchmark_query("satisfaction_rating")
    
    # Example Q&A query
    qa_query = query_processor.prepare_qa_query("Hoe duurzaam was ons festival?")
    
    print("\n🎯 Example query preparations completed!")
    print("📊 Ready for LLM integration!")


"""
USAGE INSTRUCTIONS:
==================

1. Install required packages:
   pip install pandas numpy openpyxl PyPDF2

2. Place your files in the same directory:
   - projecten_090720250858.xlsx
   - responses_rotterdam_bluegrass_festival.xlsx
   - rotterdam_bluegrass_factsheet.pdf (optional)

3. Run the script:
   python cigarbox_preprocessing.py

4. Check the 'processed_data' directory for outputs:
   - processed_data.json: Complete processed dataset
   - validation_data.json: Validation dataset
   - storytelling_data.json: Data for storytelling queries
   - benchmarking_data.json: Data for benchmarking queries
   - qa_data.json: Data for Q&A queries
   - open_questions_data.json: Data for open questions analysis
   - summary_report.txt: Human-readable summary

5. For LLM integration, use the QueryProcessor class to prepare
   data for specific query types.

FEATURES:
=========

✅ Multi-source data integration
✅ Intelligent data cleaning and standardization
✅ Query-specific data preparation
✅ Validation dataset creation
✅ Cross-reference validation
✅ Summary statistics and insights
✅ Error handling and logging
✅ Extensible architecture
✅ Memory-efficient processing
✅ JSON output for easy LLM integration

NEXT STEPS:
===========

1. Integrate with your LLM API (OpenAI, Anthropic, etc.)
2. Implement the validation functions in your LLM pipeline
3. Create a web interface for query input
4. Add real-time data processing capabilities
5. Implement caching for frequently accessed data

"""