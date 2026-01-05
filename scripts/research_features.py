"""
GPT-Researcher Script for Structural Break Detection

This script researches:
1. Advanced structural break detection features
2. Time series analysis techniques
3. Competition-winning approaches
4. Novel feature engineering methods

Usage:
    python research_features.py [topic]
    
Topics:
    - features: Research feature engineering for structural breaks
    - methods: Research detection methods and algorithms
    - winners: Research competition-winning approaches
    - statistical: Research statistical tests for change points
    - all: Research everything (default)
"""

import sys
import os
from pathlib import Path

# Check if gpt_researcher is installed
try:
    from gpt_researcher import GPTResearcher
    import asyncio
except ImportError:
    print("❌ gpt-researcher not installed")
    print("\nInstalling gpt-researcher...")
    os.system("pip install gpt-researcher")
    print("\n✅ Installation complete. Please run the script again.")
    sys.exit(0)


# Research queries
RESEARCH_QUERIES = {
    "features": """
    What are the most effective features for detecting structural breaks in time series data?
    Focus on:
    - Statistical tests (Anderson-Darling, KS test, etc.)
    - Compression-based features (Kolmogorov complexity, NCD)
    - CUSUM and change point detection features
    - Distribution comparison metrics
    - Tail behavior analysis
    - Entropy and information theory features
    
    Provide specific mathematical formulations and Python implementation guidance.
    """,
    
    "methods": """
    What are the state-of-the-art methods for structural break detection and change point detection?
    Focus on:
    - Bayesian change point detection
    - CUSUM variants (adaptive, weighted)
    - E-divisive methods
    - Binary segmentation
    - PELT (Pruned Exact Linear Time)
    - Autoencoder-based anomaly detection
    - TabPFN for tabular data
    
    Include references to recent papers and implementations.
    """,
    
    "winners": """
    What techniques and features do winning solutions use in time series competitions and structural break challenges?
    Focus on:
    - Feature engineering strategies
    - Model stacking and ensembling
    - Cross-validation strategies
    - Calibration techniques
    - Feature selection methods (RFE, importance-based)
    
    Look for Kaggle, CrunchDAO, and other ML competition solutions.
    """,
    
    "statistical": """
    What statistical tests are most effective for detecting distribution changes and structural breaks?
    Focus on:
    - Anderson-Darling test vs KS test
    - Levene's test, Bartlett's test
    - F-test for variance
    - Mann-Whitney U test
    - Cramér-von Mises test
    - Energy statistics
    - Wasserstein distance
    
    Compare power and sensitivity of different tests.
    """,
    
    "compression": """
    How can data compression be used for detecting structural breaks and measuring complexity?
    Focus on:
    - Kolmogorov complexity approximation
    - Normalized Compression Distance (NCD)
    - Compression ratio features (gzip, bz2, zlib, lzma)
    - Information content changes
    - Entropy-based features
    
    Provide practical implementation approaches for time series.
    """,
    
    "transformations": """
    What time series transformations are most effective for revealing structural breaks?
    Focus on:
    - Differencing and log transforms
    - Box-Cox transformations
    - Wavelet transforms
    - Fourier transforms
    - Hodrick-Prescott filter
    - Empirical Mode Decomposition (EMD)
    
    Explain when each transformation is most useful.
    """,
}


async def research_topic(query_name, query_text, output_dir="research_results"):
    """Research a specific topic and save results."""
    
    print(f"\n{'='*70}")
    print(f"RESEARCHING: {query_name.upper()}")
    print(f"{'='*70}\n")
    
    # Create researcher
    researcher = GPTResearcher(query=query_text, report_type="research_report")
    
    # Run research
    print("🔍 Conducting research...")
    await researcher.conduct_research()
    
    print("📝 Generating report...")
    report = await researcher.write_report()
    
    # Save report
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    report_file = output_path / f"{query_name}_research.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Research Report: {query_name.title()}\n\n")
        f.write(f"**Query:** {query_text.strip()}\n\n")
        f.write("---\n\n")
        f.write(report)
    
    print(f"✅ Report saved to: {report_file}\n")
    
    # Print summary
    print("📊 SUMMARY")
    print("="*70)
    lines = report.split('\n')
    summary_lines = [line for line in lines[:20] if line.strip()]
    for line in summary_lines:
        print(line)
    print("\n[... See full report for details ...]\n")
    
    return report


async def research_all():
    """Research all topics."""
    
    print("="*70)
    print("GPT-RESEARCHER: STRUCTURAL BREAK DETECTION")
    print("="*70)
    print()
    print(f"📚 Researching {len(RESEARCH_QUERIES)} topics...")
    print()
    
    reports = {}
    
    for query_name, query_text in RESEARCH_QUERIES.items():
        try:
            report = await research_topic(query_name, query_text)
            reports[query_name] = report
        except Exception as e:
            print(f"❌ Error researching {query_name}: {e}")
            continue
    
    # Create master summary
    print("\n" + "="*70)
    print("CREATING MASTER SUMMARY")
    print("="*70 + "\n")
    
    summary_path = Path("research_results") / "00_MASTER_SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Master Research Summary: Structural Break Detection\n\n")
        f.write(f"**Generated:** {Path.ctime(summary_path)}\n\n")
        f.write("## Topics Researched\n\n")
        
        for query_name in RESEARCH_QUERIES.keys():
            f.write(f"- [{query_name.title()}]({query_name}_research.md)\n")
        
        f.write("\n## Quick Reference\n\n")
        f.write("### Top Recommendations\n\n")
        f.write("Based on the research, here are the key takeaways:\n\n")
        
        for query_name, report in reports.items():
            f.write(f"#### {query_name.title()}\n\n")
            # Extract first few paragraphs
            lines = [line for line in report.split('\n') if line.strip()][:5]
            for line in lines:
                f.write(f"{line}\n")
            f.write("\n")
    
    print(f"✅ Master summary saved to: {summary_path}")
    print("\n" + "="*70)
    print("RESEARCH COMPLETE")
    print("="*70)
    print(f"\n📁 All reports saved to: research_results/")
    print(f"📄 Start with: research_results/00_MASTER_SUMMARY.md\n")


async def main():
    """Main entry point."""
    
    # Parse command line args
    topic = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if topic == "all":
        await research_all()
    elif topic in RESEARCH_QUERIES:
        await research_topic(topic, RESEARCH_QUERIES[topic])
    else:
        print(f"❌ Unknown topic: {topic}")
        print(f"\nAvailable topics: {', '.join(RESEARCH_QUERIES.keys())}, all")
        print(f"\nUsage: python research_features.py [topic]")
        sys.exit(1)


if __name__ == "__main__":
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("TAVILY_API_KEY"):
        print("⚠️  WARNING: No API keys found")
        print("\nGPT-Researcher requires:")
        print("  - OPENAI_API_KEY (for GPT models)")
        print("  - TAVILY_API_KEY (for web search)")
        print("\nSet them in your environment:")
        print('  $env:OPENAI_API_KEY="your-key-here"')
        print('  $env:TAVILY_API_KEY="your-key-here"')
        print()
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Run research
    asyncio.run(main())
