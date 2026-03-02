import pandas as pd

# Basic Cleaning
def basic_cleaning(df):
    df = df.copy()

    # Ensure salary is numeric (coerce errors to NaN)
    df["salary_usd"] = pd.to_numeric(df["salary_usd"], errors="coerce")

    # Drop rows with missing required_skills
    df = df.dropna(subset=["required_skills"])

    # Drop rows with missing salary
    df = df.dropna(subset=["salary_usd"])

    # Keep only positive salary values
    df = df[df["salary_usd"] > 0]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    return df

# Clean Individual Skill String
def clean_skills(skill_string):
    if not isinstance(skill_string, str):
        return []

    skills = [
        skill.strip().lower()
        for skill in skill_string.split(",")
        if skill.strip()  # remove empty strings
    ]

    return list(set(skills))  # remove duplicates

# Apply Skill Cleaning to DataFrame
def apply_skill_cleaning(df):
    df = df.copy()

    df["cleaned_skills"] = df["required_skills"].apply(clean_skills)

    # Remove rows where cleaned_skills became empty
    df = df[df["cleaned_skills"].map(len) > 0]

    df = df.reset_index(drop=True)

    return df