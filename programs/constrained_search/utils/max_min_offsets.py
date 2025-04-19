"""min/max values of original_offset_phi/theta/psi columns in CSV file."""

import argparse

import pandas as pd


def analyze_offset_angles(csv_file: str) -> None:
    """
    Print min/max values of original_offset_phi/theta/psi columns.

    Parameters
    ----------
    csv_file: str
        Path to the CSV file containing results

    Returns
    -------
    None
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Check if the required columns exist
        offset_columns = [
            "original_offset_phi",
            "original_offset_theta",
            "original_offset_psi",
        ]
        missing_columns = [col for col in offset_columns if col not in df.columns]

        if missing_columns:
            print(
                f"Error: The following required columns are missing: "
                f"{', '.join(missing_columns)}"
            )
            print(f"Available columns: {', '.join(df.columns)}")
            return

        # Print min and max for each offset column
        print(f"Analysis of angular offsets in {csv_file}:")
        print("-" * 50)

        for col in offset_columns:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            std_val = df[col].std()

            print(f"{col}:")
            print(f"  Min: {min_val:.4f}°")
            print(f"  Max: {max_val:.4f}°")
            print(f"  Range: {max_val - min_val:.4f}°")
            print(f"  Mean: {mean_val:.4f}°")
            print(f"  Std Dev: {std_val:.4f}°")
            print()

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: '{csv_file}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: '{csv_file}' is not a valid CSV file.")
    except Exception as e:
        print(f"Error: {e!s}")


def main() -> None:
    """Main function to analyze original offset angles in a results CSV file."""
    parser = argparse.ArgumentParser(
        description="Analyze original offset angles in a results CSV file."
    )
    parser.add_argument("csv_file", help="Path to the CSV file containing results")

    args = parser.parse_args()
    analyze_offset_angles(args.csv_file)


if __name__ == "__main__":
    main()
