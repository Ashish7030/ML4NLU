# Towards Multilingual LLM Evaluation for European Languages

This repository contains the statistical pipeline and datasets used to quantify the **"Multilingual Tax"** the performance disparity between High-Resource (HRL) and Medium-Resource Languages (MRL) within the EU20 benchmark framework.

## Research Overview
The analysis explores how linguistic typology and resource availability impact the zero-shot reasoning capabilities of Large Language Models. Using a dataset of 16 European languages, this implementation conducts:

1.  **Inferential Testing:** Welch’s T-test and Cohen’s $d$ to determine the significance and magnitude of the performance gap.
2.  **Computational Robustness:** Non-parametric Bootstrap resampling (n=1000) to validate findings despite small sample constraints.
3.  **Predictive Modeling:** OLS Regression to evaluate the influence of language families (Germanic, Romance, Slavic, etc.) on accuracy.

## Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Git

### Environment Setup
1. Clone the repository:
   ```bash
   git clone ([https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/Ashish7030/ML4NLU.git))
   cd YOUR_REPO_NAME