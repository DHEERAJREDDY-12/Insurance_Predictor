import argparse
import json
import os
from typing import Optional

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Batch inference for title insurance premium model")
    parser.add_argument("--model", required=True, help="Path to saved model .joblib")
    parser.add_argument("--input-csv", help="Path to input CSV with feature columns")
    parser.add_argument("--input-json", help="Inline JSON string or path to JSON file with records list")
    parser.add_argument("--output-csv", help="Where to save predictions CSV")

    args = parser.parse_args()

    model = joblib.load(args.model)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
    elif args.input_json:
        if os.path.exists(args.input_json):
            with open(args.input_json, "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            records = json.loads(args.input_json)
        df = pd.DataFrame.from_records(records)
    else:
        raise ValueError("Provide --input-csv or --input-json")

    preds = model.predict(df)
    out = df.copy()
    out["prediction"] = preds

    if args.output_csv:
        # Only create directory if output_csv has a directory path
        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        out.to_csv(args.output_csv, index=False)
        print(f"Saved predictions to {args.output_csv}")
    else:
        print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
