#!/usr/bin/env python3
"""
Main Results Processing Script

Processes MMLU evaluation results and generates comprehensive analysis.
Handles CSV consolidation, performance analysis, and report generation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add processor module to path
sys.path.append(os.path.dirname(__file__))

from processor import ResultConsolidator, PerformanceAnalyzer, ReportGenerator


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(
        description='Process MMLU evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all results with default settings
  python process_results.py
  
  # Process specific results directory with verbose output
  python process_results.py --results-dir ./custom_results --output-dir ./analysis --verbose
  
  # Generate only consolidation without analysis
  python process_results.py --consolidate-only
  
  # Generate only analysis plots
  python process_results.py --analysis-only --output-dir ./plots
        """
    )
    
    parser.add_argument(
        '--results-dir', 
        default='./results',
        help='Directory containing evaluation results (default: ./results)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./processed_results',
        help='Output directory for processed results (default: ./processed_results)'
    )
    
    parser.add_argument(
        '--consolidate-only',
        action='store_true',
        help='Only perform CSV consolidation, skip analysis and reports'
    )
    
    parser.add_argument(
        '--analysis-only',
        action='store_true',
        help='Only perform analysis (requires existing consolidated results)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation (faster processing)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate directories
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔄 MMLU Results Processor")
    print("="*50)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Processing mode: {'Consolidate only' if args.consolidate_only else 'Analysis only' if args.analysis_only else 'Full pipeline'}")
    print("="*50)
    
    try:
        if not args.analysis_only:
            # Step 1: Consolidate results
            logger.info("Starting result consolidation...")
            print("\n📊 Step 1: Consolidating Results")
            
            consolidator = ResultConsolidator(str(results_dir))
            consolidated_results = consolidator.process_all(str(output_dir))
            
            if not consolidated_results.models:
                logger.error("No models found to process. Check your results directory structure.")
                print("❌ No models found. Check your results directory structure.")
                sys.exit(1)
            
            print(f"✅ Consolidated {len(consolidated_results.models)} models")
            
            if args.consolidate_only:
                print("\n🎉 Consolidation complete!")
                sys.exit(0)
        
        else:
            # Load existing consolidated results for analysis-only mode
            logger.info("Analysis-only mode: looking for existing consolidated results...")
            # This would require implementing a load mechanism
            print("❌ Analysis-only mode requires existing consolidated results (not yet implemented)")
            sys.exit(1)
        
        if not args.consolidate_only:
            # Step 2: Performance Analysis
            logger.info("Starting performance analysis...")
            print("\n📈 Step 2: Performance Analysis")
            
            analyzer = PerformanceAnalyzer(consolidated_results)
            
            analysis_dir = output_dir / "analysis"
            analyzer.generate_analysis_report(str(analysis_dir))
            
            print("✅ Performance analysis complete")
            
            # Step 3: Report Generation
            logger.info("Starting report generation...")
            print("\n📋 Step 3: Report Generation")
            
            report_generator = ReportGenerator(consolidated_results)
            
            reports_dir = output_dir / "reports"
            report_generator.generate_all_reports(str(reports_dir))
            
            print("✅ Report generation complete")
        
        # Summary
        print("\n🎉 Processing Complete!")
        print("="*50)
        print(f"📁 All outputs saved to: {output_dir}")
        print("\nGenerated files:")
        print("  📊 Consolidated CSVs and Excel files")
        if not args.consolidate_only:
            print("  📈 Performance analysis and plots")
            print("  📋 Executive summary and detailed reports")
        print("\n💡 Check the output directory for all generated files.")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"\n❌ Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
