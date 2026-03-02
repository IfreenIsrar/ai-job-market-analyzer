from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Model Saving
def save_model(model, path="outputs/model/salary_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")

# Skill Frequency Analysis
def get_skill_frequency(df):
    all_skills = [
        skill
        for skills_list in df["cleaned_skills"]
        for skill in skills_list
    ]

    return Counter(all_skills)

# Cluster Interpretation
def top_skills_per_cluster(model, mlb, top_n=10):
    skill_names = mlb.classes_
    sorted_centroids = model.cluster_centers_.argsort()[:, ::-1]

    for cluster_id in range(model.n_clusters):
        print(f"\nCluster {cluster_id} Top Skills:")
        for index in sorted_centroids[cluster_id, :top_n]:
            print(f"  - {skill_names[index]}")

# Salary Prediction Model
def salary_prediction(df, skill_df):
    """
    Train RandomForest model to predict salary.
    Uses structured + skill features.
    """
    df = df.copy()
    # Encode experience level safely
    df["experience_level_encoded"] = df["experience_level"].map({
        "EN": 0,
        "MI": 1,
        "SE": 2,
        "EX": 3
    })
    # Structured numeric features
    numeric_features = [
        "experience_level_encoded",
        "years_experience",
        "remote_ratio",
        "benefits_score"
    ]

    X_numeric = df[numeric_features]
    # Combine structured + skill features
    X = pd.concat(
        [X_numeric.reset_index(drop=True),
         skill_df.reset_index(drop=True)],
        axis=1
    )

    y = df["salary_usd"]
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1  # faster training
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel Performance")
    print("-" * 30)
    print(f"MAE: {mae:.2f}")
    print(f"R² : {r2:.4f}")

    # Feature importance
    importance_df = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        })
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nTop 10 Most Important Features")
    print("-" * 30)
    print(importance_df.head(10))

    return model, importance_df