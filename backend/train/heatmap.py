import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Generate transaction heatmap for a selected user."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to engineered_transactions.csv"
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="User ID (cc_num) to visualize"
    )
    parser.add_argument(
        "--output",
        default="transaction_heatmap.png",
        help="Output image file"
    )
    parser.add_argument(
        "--fraud-only",
        action="store_true",
        help="Plot only fraud transactions"
    )

    args = parser.parse_args()

    # Load engineered dataset
    df = pd.read_csv(args.input)
    df["user_id"] = df["user_id"].astype(str)

    # Filter by user
    user_df = df[df["user_id"] == str(args.user_id)].copy()

    if user_df.empty:
        raise ValueError(f"No data found for user_id={args.user_id}")

    # Fraud-only option
    if args.fraud_only:
        user_df = user_df[user_df["is_fraud"] == 1]
        if user_df.empty:
            raise ValueError("No fraud transactions found for this user.")

    legit_df = user_df[user_df["is_fraud"] == 0]
    fraud_df = user_df[user_df["is_fraud"] == 1]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Heatmap (legit)
    if not legit_df.empty:
        hb = ax.hexbin(
            legit_df["transaction_lng"],
            legit_df["transaction_lat"],
            gridsize=30,
            mincnt=1
        )
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("Legit transaction density")

    # Fraud overlay
    if not fraud_df.empty:
        ax.scatter(
            fraud_df["transaction_lng"],
            fraud_df["transaction_lat"],
            marker="x",
            s=70,
            label="Fraud transactions"
        )

    ax.set_title(f"Transaction Heatmap (User {args.user_id})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if not fraud_df.empty:
        ax.legend()

    fig.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)

    print(f"Saved heatmap → {args.output}")


if __name__ == "__main__":
    main()