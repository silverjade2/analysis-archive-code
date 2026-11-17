# Labeling instructions

You label job postings by the kinds of work the role actually does. Postings may be in Korean or English.
Choose ALL task types that are a substantial part of the role's responsibilities. Ignore company introductions, benefits, hiring steps, and skills that are only nice-to-have. Also choose the single primary task. If the posting is not a data, analytics, or AI role at all, return an empty tasks list and primary_task 'none'.

Task types:
- reporting_dashboard: Build and maintain recurring reports and BI dashboards
- metric_design: Define core metrics, KPI frameworks and event tracking
- experimentation_causal: Design and analyze A/B tests and estimate causal effects
- business_insight: Ad-hoc analysis that turns data into insights and recommendations for business decisions
- customer_marketing: Customer segmentation, campaign performance, LTV and churn analysis
- predictive_modeling: Build predictive, classification or scoring models
- recommendation_search: Recommendation, search, ranking and personalization models
- optimization_operations: Optimization and simulation for operational decisions such as pricing, inventory and logistics
- data_pipeline: Build ETL/ELT pipelines, data marts and data models
- data_governance: Data quality, governance, metadata and catalog management
- ml_engineering: Deploy, serve and monitor ML models and run training pipelines
- llm_agent_dev: Build and evaluate LLM applications, RAG systems and AI agents
- ai_adoption_automation: Identify and drive AI adoption and workflow automation in business processes
- research: Research new models or methodologies, publish papers

Boundary rules (decided earlier, apply exactly):
- Jobs that rate/evaluate content or search results for AI training, or annotate/label data for AI training
  (part-time raters, labeling interns, legal data annotation): primary_task 'none', tasks [].
- Professor / faculty / lecturer positions at universities, whatever research they mention: primary_task 'none', tasks [].
  (A company that hires PhDs for paper consulting is NOT a faculty post: label its work normally, e.g. research.)

Output format: one JSON object per line (JSONL), nothing else, one line per posting, in the order read:
{"doc_id": "<exact doc_id from the file>", "tasks": ["<task id>", ...], "primary_task": "<task id or none>"}
- Use only the task ids listed above. primary_task must be one of tasks (or 'none' with an empty tasks list).
- Do not add explanations. Do not skip any posting.
