from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def encode_skills(df):
    """
    Convert cleaned_skills list into multi-hot encoded matrix.
    Each skill becomes a binary column (0 or 1).
    """

    mlb = MultiLabelBinarizer()
    skill_matrix = mlb.fit_transform(df["cleaned_skills"])

    skill_df = pd.DataFrame(skill_matrix, columns=mlb.classes_)

    return skill_df, mlb