from src.data_collection import load_data
from src.preprocessing import basic_cleaning, apply_skill_cleaning
from src.analysis import (
    get_skill_frequency,
    top_skills_per_cluster,
    salary_prediction,
    save_model
)
from src.skill_extraction import encode_skills
from src.clustering import elbow_method, cluster_jobs
from src.visualization import (
    salary_by_experience,
    salary_distribution,
    plot_top_skills,
    plot_feature_importance,
    salary_by_cluster
)
import pandas as pd
import os
#  Setup Directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# Load Raw Data
df = load_data("data/raw/ai_job_dataset.csv")
print("Initial shape:", df.shape)

# Basic Cleaning
df = basic_cleaning(df)
print("After basic cleaning:", df.shape)

# Skill Cleaning
df = apply_skill_cleaning(df)

print("\nSample cleaned skills:")
print(df["cleaned_skills"].head())

# Skill Frequency Analysis
skill_counts = get_skill_frequency(df)

print("\nTotal unique skills:", len(skill_counts))
print("Top 20 skills:")
print(skill_counts.most_common(20))

# Save & plot top skills
plot_top_skills(skill_counts)

top_skills_df = pd.DataFrame(
    skill_counts.most_common(50),
    columns=["skill", "count"]
)
top_skills_df.to_csv("outputs/reports/top_skills.csv", index=False)

# Encode Skills
skill_df, mlb = encode_skills(df)

print("\nEncoded skill shape:", skill_df.shape)

# Elbow Method (Cluster Selection)
elbow_method(skill_df)

# Clustering
labels, cluster_model = cluster_jobs(skill_df, k=4)
df["cluster"] = labels

print("\nCluster distribution:")
print(df["cluster"].value_counts())

top_skills_per_cluster(cluster_model, mlb)

salary_by_cluster(df)

df["cluster"].value_counts().to_csv(
    "outputs/reports/cluster_distribution.csv"
)

# Salary Prediction
model, importance_df = salary_prediction(df, skill_df)

# Save trained model
save_model(model)

# Save feature importance CSV
importance_df.to_csv(
    "outputs/reports/feature_importance.csv",
    index=False
)

# Visualizations
salary_by_experience(df)
salary_distribution(df)

plot_feature_importance(model, importance_df["feature"])

# Save Processed Dataset
df.to_csv(
    "data/processed/final_dataset_with_clusters.csv",
    index=False
)


print("\nPipeline execution completed successfully.")