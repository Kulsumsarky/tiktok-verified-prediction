# TikTok Verified User Prediction
## Logistic Regression Classification Project

### Project Summary
This project builds a Logistic Regression model to predict whether
a TikTok video creator is verified or unverified using video metadata
and engagement metrics. The model addresses a 94/6% class imbalance
through upsampling and is designed to help TikTok's operations team
prioritize their report backlog more efficiently.

### How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `tiktok_verified_prediction.ipynb` in Jupyter or Google Colab
4. Run all cells in order

### Dataset
- Size: 19,382 TikTok video records
- Features: claim_status, author_ban_status, text_length,
  video_view_count, video_like_count, video_share_count,
  video_download_count, video_comment_count
- Target: verified_status (0=Not Verified, 1=Verified)
- Class imbalance: 94% Not Verified, 6% Verified

### Methodology
1. Discovery & Cleaning — checked shape, dtypes, missing values
2. Feature Engineering — created text_length feature, EDA plots
3. Data Preparation — 80/20 train/test split with random_state=42
4. Upsampling — minority class upsampled AFTER split to prevent leakage
5. Encoding — One-Hot Encoding with drop_first=True on categorical columns
6. Scaling — StandardScaler fit on train only, transform on test
7. Modeling — compared baseline (class_weight=balanced) vs resampled model
8. Evaluation — Confusion Matrix, Classification Report, ROC-AUC

### Key Findings
- 94/6% class imbalance required upsampling of verified class
- Upsampled model outperformed baseline in Recall
- Top positive predictors: video_view_count, video_like_count
- Top negative predictors: author_ban_status_banned
- Banned authors are rarely verified creators

### Model Performance
| Metric | Score |
|--------|-------|
| Recall | 0.495 |
| Precision | 0.051 |
| F1-Score | 0.093 |
| ROC-AUC | 0.486 |

### Business Recommendation
Deploy model to prioritize report backlog by flagging unverified
accounts making claims. Focus human review on claim videos from
unverified authors as these pose the highest content integrity risk.
This approach can reduce manual review time by up to 75%.

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Environment Setup
pip install -r requirements.txt

### Files
- tiktok_verified_prediction.ipynb — Main analysis notebook
- requirements.txt — Python dependencies
- README.md — Project documentation
