# AI Assistant Usage in Student Life: Final Outcome Prediction

**Author:** 0% Accuracy  
**Competition:** CompIT 2025 by Ristek HMIT

## 1. Introduction and Project Goal

This project aims to analyze and model the session behavior of students using an AI Assistant. The primary objective is to predict the `FinalOutcome` of a user's session based on a synthetic dataset mimicking real-world interactions.

The approach involves a structured machine learning workflow:

1.  **Exploratory Data Analysis (EDA):** To understand data distributions, variable relationships, and gain initial insights.
2.  **Preprocessing & Feature Engineering:** To clean the data and create new, informative features that enhance model performance.
3.  **Modeling:** To train a robust LightGBM classification model using cross-validation.
4.  **Evaluation:** To analyze the model's performance and the importance of each feature.

## 2. Exploratory Data Analysis (EDA) Highlights

Key insights from the EDA phase shaped the entire modeling strategy:

- **Imbalanced Target Variable:** The `FinalOutcome` classes were imbalanced, with `Assignment Completed` being the majority class and `Gave Up` the minority. This confirmed the need for strategies like stratified sampling and class weighting.
- **Right-Skewed Numerical Features:** `SessionLengthMin` and `TotalPrompts` showed a right-skewed distribution, indicating that a logarithmic transformation could help normalize them.
- **Temporal Trends:** A clear correlation was found between session frequency and the academic calendar. Usage peaked during mid-term and final exam periods and dropped significantly during holidays, confirming the importance of date-based features.
- **Categorical Insights:** `UsedAgain` showed a strong correlation with positive outcomes, suggesting that user retention is a good indicator of success.

## 3. Preprocessing & Feature Engineering

This was the most critical phase for improving model performance. The following techniques were applied:

| #   | Feature Type                      | Technique & Rationale                                                                                                                                                                                                                                                                                                                                                                        |
| --- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Temporal Features**             | Extracted `day_of_week`, `day_of_year`, `month`, and `is_weekend` from `SessionDate` to capture academic cycle patterns.                                                                                                                                                                                                                                                                     |
| 2   | **Interaction Features**          | Created `prompts_per_minute` to measure session intensity and `satisfaction_gap` (Satisfaction Rating - AI Assistance Level) to quantify the gap between perceived help and user satisfaction.                                                                                                                                                                                               |
| 3   | **Log Transformation**            | Applied `log1p` to `SessionLengthMin` and `TotalPrompts` to normalize their skewed distributions, making it easier for the model to learn.                                                                                                                                                                                                                                                   |
| 4   | **Aggregate & Relative Features** | This was the key to unlocking performance. **Aggregate features** (`mean`, `std`, `max`) were calculated for categorical groups (`Discipline`, `TaskType`, `StudentLevel`) to establish a baseline. **Relative features** (differences, ratios, and z-scores) were then created to measure how much a single session deviated from its group average, providing powerful contextual signals. |
| 5   | **Ordinal & One-Hot Encoding**    | `StudentLevel` was ordinally encoded to preserve its inherent order. `Discipline` and `TaskType` were one-hot encoded as they have no natural order.                                                                                                                                                                                                                                         |

## 4. Modeling and Evaluation

### Model Selection

- **LightGBM (LGBM) Classifier:** Chosen for its high performance, speed, and efficiency with large tabular datasets.
- **Key Parameter:** `class_weight='balanced'` was used to address the class imbalance issue.

### Validation Strategy

- **Stratified K-Fold Cross-Validation (5 Folds):** This robust technique ensures that the proportion of target classes is maintained in each fold, leading to a reliable and unbiased performance estimate.
- **Early Stopping:** Used within each fold to prevent overfitting and find the optimal number of training iterations.

### Performance

- The model achieved a stable and solid average **Macro F1 Score of approximately 0.38**.
- The low standard deviation across folds indicates that the model is consistent and generalizes well.

## 5. Feature Importance Analysis

The feature importance analysis validated the entire engineering approach. The most influential features were not the original ones but the **custom-built features**, including:

1.  `day_of_year` (Temporal)
2.  `prompts_per_min_vs_student_avg` (Relative)
3.  `prompts_per_minute` (Interaction)
4.  `satisfaction_gap` (Interaction)
5.  `session_vs_student_avg` (Relative)

This proves that providing the model with **contextual information** how a session compares to a group average was the most critical factor for success.

## 6. Final Conclusion

This project successfully developed a robust end-to-end machine learning pipeline. The key takeaway is the immense value of **context-driven feature engineering**, which proved to be far more impactful than raw data alone. The stable and consistent evaluation results demonstrate the reliability of the final model.

### Future Improvements

- **Hyperparameter Tuning:** Use automated tools like Optuna to find the optimal set of model parameters.
- **Explore Other Models:** Experiment with XGBoost, CatBoost, or ensemble methods.
- **Advanced Feature Engineering:** Create dynamic features based on session sequences (e.g., lag features) to capture user behavior changes over time.
