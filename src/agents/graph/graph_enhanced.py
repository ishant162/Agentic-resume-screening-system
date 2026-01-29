"""
Enhanced Resume Screening Agent - Agentic Workflow with Tools

This is the super-agentic version with:
- Tool Coordinator (LLM decides which tools to use)
- Company Verification (Web Search)
- GitHub Analysis
- Skill Taxonomy (Semantic matching)
- Quality Checker (Self-reflection)
- Salary Estimator
- ATS Scorer
- Bias Detector
"""

from typing import Literal

from langgraph.graph import END, StateGraph

# Import all nodes
from src.agents.nodes import (
    ats_scorer_node,
    bias_detector_node,
    candidate_enricher_node,
    education_verifier_node,
    experience_analyzer_enhanced_node,
    # Original nodes
    job_analyzer_node,
    quality_checker_node,
    question_generator_node,
    report_generator_node,
    resume_parser_node,
    salary_estimator_node,
    scorer_node,
    skill_matcher_enhanced_node,
    # Enhanced agentic nodes
    tool_coordinator_node,
)
from src.state.state import AgentState


def should_reanalyze(state: dict) -> Literal["reanalyze", "continue"]:
    """
    Conditional edge: Decide if we need to re-analyze

    This is the self-reflection mechanism.
    If quality checker determines low confidence, loop back.
    """
    quality_check = state.get("quality_check", {})

    needs_reanalysis = quality_check.get("needs_reanalysis", False)
    reanalysis_count = state.get("reanalysis_count", 0)

    # Limit reanalysis to prevent infinite loops
    if needs_reanalysis and reanalysis_count < 2:
        print("\n" + "="*80)
        print("🔄 QUALITY CHECKER TRIGGERED RE-ANALYSIS")
        print("="*80)
        print(f"Confidence was too low. Re-analyzing (attempt {reanalysis_count + 1}/2)...\n")
        return "reanalyze"
    else:
        if needs_reanalysis and reanalysis_count >= 2:
            print("\n⚠️  Maximum re-analysis attempts reached. Continuing with current results.\n")
        return "continue"


def create_enhanced_screening_graph() -> StateGraph:
    """
    Create the enhanced agentic resume screening workflow

    New Features:
    - Tool Coordinator: LLM decides which tools to use
    - Candidate Enrichment: Web search, GitHub, etc.
    - Enhanced Matching: Semantic skill understanding
    - Quality Checker: Self-reflection and confidence validation
    - Additional Analyses: Bias detection, salary estimation, ATS scoring

    Workflow:
        START
          ↓
        Job Analyzer
          ↓
        Resume Parser
          ↓
        Tool Coordinator (LLM decides tools) ← AGENTIC
          ↓
        Candidate Enricher (runs tools)
          ↓
        Enhanced Skill Matcher (uses taxonomy)
          ↓
        Enhanced Experience Analyzer (uses company data)
          ↓
        Education Verifier
          ↓
        Scorer & Ranker
          ↓
        Quality Checker (self-reflection) ← AGENTIC
          ↓
        [Conditional: Re-analyze if needed OR Continue]
          ↓
        Bias Detector
          ↓
        Salary Estimator
          ↓
        ATS Scorer
          ↓
        Report Generator
          ↓
        Question Generator
          ↓
        END

    Returns:
        Compiled StateGraph ready for execution
    """

    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)

    print("🔧 Building enhanced agentic graph...")

    # ============================================================
    # PHASE 1: Initial Analysis
    # ============================================================

    workflow.add_node("job_analyzer", job_analyzer_node)
    workflow.add_node("resume_parser", resume_parser_node)

    # ============================================================
    # PHASE 2: Agentic Tool Selection & Enrichment (NEW)
    # ============================================================

    workflow.add_node("tool_coordinator", tool_coordinator_node)
    workflow.add_node("candidate_enricher", candidate_enricher_node)

    # ============================================================
    # PHASE 3: Enhanced Analysis with Tool Data
    # ============================================================

    workflow.add_node("skill_matcher_enhanced", skill_matcher_enhanced_node)
    workflow.add_node("experience_analyzer_enhanced", experience_analyzer_enhanced_node)
    workflow.add_node("education_verifier", education_verifier_node)
    workflow.add_node("scorer", scorer_node)

    # ============================================================
    # PHASE 4: Quality Control & Self-Reflection (NEW)
    # ============================================================

    workflow.add_node("quality_checker", quality_checker_node)

    # ============================================================
    # PHASE 5: Additional Analyses (NEW)
    # ============================================================

    workflow.add_node("bias_detector", bias_detector_node)
    workflow.add_node("salary_estimator", salary_estimator_node)
    workflow.add_node("ats_scorer", ats_scorer_node)

    # ============================================================
    # PHASE 6: Output Generation
    # ============================================================

    workflow.add_node("report_generator", report_generator_node)
    workflow.add_node("question_generator", question_generator_node)

    # ============================================================
    # DEFINE WORKFLOW EDGES
    # ============================================================

    print("  ✅ Added all nodes")
    print("  🔗 Connecting workflow edges...")

    # Start with job analysis
    workflow.set_entry_point("job_analyzer")

    # Phase 1: Initial parsing
    workflow.add_edge("job_analyzer", "resume_parser")

    # Phase 2: Agentic tool selection
    workflow.add_edge("resume_parser", "tool_coordinator")
    workflow.add_edge("tool_coordinator", "candidate_enricher")

    # Phase 3: Enhanced analysis
    workflow.add_edge("candidate_enricher", "skill_matcher_enhanced")
    workflow.add_edge("skill_matcher_enhanced", "experience_analyzer_enhanced")
    workflow.add_edge("experience_analyzer_enhanced", "education_verifier")
    workflow.add_edge("education_verifier", "scorer")

    # Phase 4: Quality check with conditional routing
    workflow.add_edge("scorer", "quality_checker")

    # Conditional: Re-analyze if needed or continue
    workflow.add_conditional_edges(
        "quality_checker",
        should_reanalyze,
        {
            "reanalyze": "experience_analyzer_enhanced",  # Loop back for deeper analysis
            "continue": "bias_detector"  # Continue forward
        }
    )

    # Phase 5: Additional analyses
    workflow.add_edge("bias_detector", "salary_estimator")
    workflow.add_edge("salary_estimator", "ats_scorer")

    # Phase 6: Output generation
    workflow.add_edge("ats_scorer", "report_generator")
    workflow.add_edge("report_generator", "question_generator")

    # End
    workflow.add_edge("question_generator", END)

    print("  ✅ Workflow edges connected")

    # Compile the graph
    print("  ⚙️  Compiling graph...")
    app = workflow.compile()

    print("✅ Enhanced agentic graph created successfully!\n")

    return app


def visualize_enhanced_graph(output_path: str = "enhanced_workflow_diagram.png"):
    """Visualize the enhanced workflow graph"""
    try:
        app = create_enhanced_screening_graph()
        graph_image = app.get_graph().draw_mermaid_png()

        with open(output_path, "wb") as f:
            f.write(graph_image)

        print(f"✅ Enhanced graph visualization saved to {output_path}")

    except Exception as e:
        print(f"⚠️  Could not generate graph visualization: {e}")
        print("(This is optional - requires graphviz)")


def print_graph_summary():
    """Print a summary of the enhanced graph structure"""
    print("\n" + "="*80)
    print("ENHANCED AGENTIC WORKFLOW STRUCTURE")
    print("="*80)

    print("\n📋 PHASE 1: Initial Analysis")
    print("  1. Job Analyzer - Extract requirements from JD")
    print("  2. Resume Parser - Parse all PDF resumes")

    print("\n🧠 PHASE 2: Agentic Tool Selection & Enrichment")
    print("  3. Tool Coordinator - LLM decides which tools to use per candidate")
    print("  4. Candidate Enricher - Execute selected tools:")
    print("     • Web Search (Company Verification)")
    print("     • GitHub Analyzer (Skill Validation)")
    print("     • Skill Taxonomy (Semantic Matching)")

    print("\n🔍 PHASE 3: Enhanced Analysis")
    print("  5. Skill Matcher Enhanced - Uses semantic taxonomy")
    print("  6. Experience Analyzer Enhanced - Uses company data")
    print("  7. Education Verifier")
    print("  8. Scorer & Ranker")

    print("\n✅ PHASE 4: Quality Control & Self-Reflection")
    print("  9. Quality Checker - Reviews analysis confidence")
    print("     ├─→ [Low Confidence] Loop back to step 6 (re-analyze)")
    print("     └─→ [High Confidence] Continue to next phase")

    print("\n📊 PHASE 5: Additional Analyses")
    print("  10. Bias Detector - Flags potential hiring biases")
    print("  11. Salary Estimator - Compensation recommendations")
    print("  12. ATS Scorer - Resume optimization score")

    print("\n📄 PHASE 6: Output Generation")
    print("  13. Report Generator - Comprehensive markdown report")
    print("  14. Question Generator - Personalized interview questions")

    print("\n" + "="*80)
    print("KEY AGENTIC FEATURES:")
    print("="*80)
    print("  ✨ Tool Coordinator: LLM intelligently selects tools per candidate")
    print("  ✨ Conditional Routing: Self-reflection can trigger re-analysis")
    print("  ✨ Semantic Understanding: Skill taxonomy for better matching")
    print("  ✨ Multi-Tool Enrichment: Web search + GitHub + taxonomy")
    print("  ✨ Quality Assurance: Confidence validation with feedback loop")
    print("  ✨ Fairness: Bias detection for ethical hiring")
    print("  ✨ Comprehensive: Salary + ATS scoring for complete assessment")
    print("="*80 + "\n")


# Test the graph
if __name__ == "__main__":
    print("="*80)
    print("ENHANCED AGENTIC RESUME SCREENING GRAPH")
    print("="*80 + "\n")

    print_graph_summary()

    print("Creating enhanced graph...")
    app = create_enhanced_screening_graph()

    print("\n" + "="*80)
    print("GRAPH STATISTICS")
    print("="*80)
    print("  Total Nodes: 14")
    print("  Conditional Edges: 1 (quality checker re-analysis)")
    print("  Linear Edges: 12")
    print("  Entry Point: job_analyzer")
    print("  Exit Point: question_generator")
    print("="*80 + "\n")

    print("Attempting to create workflow diagram...")
    visualize_enhanced_graph()

    print("\n✅ Enhanced graph is ready for execution!")
    print("\nTo run the agent:")
    print("  python scripts/run_enhanced_agent.py --job <job.txt> --resumes <resumes/*.pdf>")
