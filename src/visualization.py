import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Global Styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# Ensure output directory exists
os.makedirs("outputs/figures", exist_ok=True)

# Salary by Experience Level
def salary_by_experience(df):

    avg_salary = (
        df.groupby("experience_level")["salary_usd"]
        .mean()
        .sort_index()
        .reset_index()
    )

    plt.figure()
    sns.barplot(
        data=avg_salary,
        x="experience_level",
        y="salary_usd",
        hue="experience_level",
        palette="viridis",
        legend=False
    )

    plt.title("Average Salary by Experience Level")
    plt.xlabel("Experience Level")
    plt.ylabel("Average Salary (USD)")

    plt.tight_layout()
    plt.savefig("outputs/figures/salary_by_experience.png")
    plt.close()

# Salary Distribution
def salary_distribution(df):

    plt.figure()
    sns.histplot(
        data=df,
        x="salary_usd",
        bins=30,
        kde=True,
        color="#2E86C1"
    )

    plt.title("Salary Distribution")
    plt.xlabel("Salary (USD)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("outputs/figures/salary_distribution.png")
    plt.close()


# Top Skills Plot
def plot_top_skills(skill_counts):

    top = skill_counts.most_common(15)
    skills, counts = zip(*top)

    df_plot = pd.DataFrame({
        "skill": skills,
        "count": counts
    })

    plt.figure()
    sns.barplot(
        data=df_plot,
        x="count",
        y="skill",
        hue="skill",
        palette="magma",
        legend=False
    )

    plt.title("Top 15 Most Demanded Skills")
    plt.xlabel("Job Count")
    plt.ylabel("Skill")

    plt.tight_layout()
    plt.savefig("outputs/figures/top_skills.png")
    plt.close()

# Feature Importance Plot
def plot_feature_importance(model, feature_names):

    importances = model.feature_importances_

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    df_importance = (
        df_importance
        .sort_values(by="importance", ascending=False)
        .head(15)
    )

    plt.figure()
    sns.barplot(
        data=df_importance,
        x="importance",
        y="feature",
        hue="feature",
        palette="coolwarm",
        legend=False
    )

    plt.title("Top Salary Influencing Features")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.savefig("outputs/figures/feature_importance.png")
    plt.close()


# Salary by Cluster
def salary_by_cluster(df):

    avg_salary = (
        df.groupby("cluster")["salary_usd"]
        .mean()
        .reset_index()
    )

    plt.figure()
    sns.barplot(
        data=avg_salary,
        x="cluster",
        y="salary_usd",
        hue="cluster",
        palette="cubehelix",
        legend=False
    )

    plt.title("Average Salary by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Average Salary (USD)")

    plt.tight_layout()
    plt.savefig("outputs/figures/salary_by_cluster.png")
    plt.close()