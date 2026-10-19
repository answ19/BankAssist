BankAssist
An evidence-grounded banking customer-service assistant.

Course: CECS 551 - Advanced Artificial Intelligence
Team: TriNova
Current phase: Phase 3 - Preliminary Data Analysis and Feasibility Analysis
Overview
BankAssist will classify customer banking questions and return a reviewed FAQ response. When confidence is low or no approved answer is available, it will recommend human assistance.
Workflow
Customer query → preprocessing → intent classification → confidence check → FAQ response or escalation.
Team
Member	Responsibilities
Anuja Sawant -	Preprocessing, shared data splits, baseline model and integration
Sakshi Tejwani	- Architecture, FAQ lookup and confidence-based fallback
Suraj Papanna	Dataset - inspection, EDA and DistilBERT experiments
Kashish Jethmalani - Evaluation, error analysis and feasibility assessment


All members contribute code, documentation and peer review.
Dataset
We plan to use BANKING77 from PolyAI. FAQ answers will be prepared and reviewed separately.
Planned Tools
Python, Google Colab/Jupyter, pandas, Matplotlib, scikit-learn, PyTorch and Hugging Face Transformers.
- Baseline: TF-IDF + Logistic Regression.
- Proposed deep model: DistilBERT.
  
Project Folders
- notebooks/: Data analysis and model experiments.
- src/: Reusable processing, prediction and evaluation code.
- data/: Data-access instructions, shared splits and reviewed FAQs.
- results/: Figures, metrics and experiment summaries.
- docs/: Reports and architecture documentation.
  
Working Guidelines
Use the same saved data splits and label mapping across models. Fit learned preprocessing only on training data. Reserve the official test set for final evaluation. Record experiment settings and report only verified results.
Status
Initial development. Setup instructions and results will be added as working components are completed. The prototype will not access accounts, perform transactions or provide personalized financial advice.
