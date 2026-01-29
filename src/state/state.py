import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State that flows through the agent graph"""

    # === INPUTS ===
    job_description: str  # Raw job description text
    resumes: list[bytes]  # List of PDF resume files as bytes
    resume_filenames: list[str] | None  # Original filenames

    # === PROCESSED JOB INFO ===
    # Parsed job requirements (will be JobRequirements model)
    job_requirements: dict | None

    # === PROCESSED CANDIDATES ===
    candidates: Annotated[list[dict], operator.add]

    # === SCORING RESULTS ===
    skill_scores: list[dict] | None  # SkillScore for each candidate
    # ExperienceScore for each candidate
    experience_scores: list[dict] | None
    education_scores: list[dict] | None  # EducationScore for each candidate

    # === FINAL OUTPUTS ===
    candidate_scores: list[dict] | None  # CandidateScore for each candidate
    ranked_candidates: list[dict] | None  # RankedCandidate list (sorted)
    report: str | None  # Final markdown report
    # {candidate_name: [questions]}
    interview_questions: dict[str, list[str]] | None

    # === INTERACTIVE Q&A ===
    user_question: str | None  # User's follow-up question
    agent_response: str | None  # Agent's response to question
    # Chat history
    conversation_history: Annotated[list[BaseMessage], operator.add]

    # === METADATA ===
    current_step: str | None  # Track which node we're in
    errors: Annotated[list[str], operator.add]  # Collect any errors

    # NEW fields for enhanced workflow:
    tool_plan: dict | None  # ✅
    company_verifications: dict | None  # ✅
    github_analyses: dict | None  # ✅
    skill_taxonomy_data: dict | None  # ✅
    quality_check: dict | None  # ✅
    reanalysis_count: int | None  # ✅
    bias_analysis: dict | None  # ✅
    salary_estimates: dict | None  # ✅
    ats_scores: dict | None  # ✅
