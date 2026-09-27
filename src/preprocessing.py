import pandas as pd

# Path to the 15-row dataset
DATA_PATH = "data/V2V_Small_Explainable_Dataset.xlsx"


def load_dataset():

    # Load Excel dataset
    df = pd.read_excel(DATA_PATH)

    # Remove accidental spaces from column names
    df.columns = df.columns.str.strip()

    print("=" * 60)
    print("V2V SAFETY MESSAGE DATASET")
    print("=" * 60)

    print("\nDataset loaded successfully!")
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))

    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    print("\nFirst 5 rows:")
    print(df.head())

    # Check criticality column
    if "criticality" in df.columns:
        print("\nCriticality distribution:")
        print(df["criticality"].value_counts())

    # Find the deadline column automatically
    deadline_columns = [
        col for col in df.columns
        if "deadline" in col.lower()
    ]

    if deadline_columns:
        deadline_column = deadline_columns[0]

        print("\nDeadline column found:", deadline_column)
        print("\nDeadline statistics:")
        print(df[deadline_column].describe())
    else:
        print("\nNo deadline column was found.")

    return df


if __name__ == "__main__":
    df = load_dataset()