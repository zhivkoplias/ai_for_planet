#!/usr/bin/env python3
"""
Parse OpenAlex-style output file into a CSV of (openalexid, key, value).

Usage (example):
    python3 04-parse-llm-output.py input_file.out output.csv

The script defines a single main function `parse_text_to_csv` and also
provides a simple CLI when executed as a script.
"""

# ---- Standard library imports -----------------------------------------------
import re
import ast
import sys
from typing import Optional

# ---- Third-party imports ----------------------------------------------------
import pandas as pd

# ---- NLTK downloads (only when needed) --------------------------------------
# Downloads are commented out by default to avoid repeated network calls.
# Uncomment and run once if the required corpora/models are missing.
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# ---- Main parsing function --------------------------------------------------
def parse_text_to_csv(input_file: str, output_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Parse a text file containing repeated blocks separated by the delimiter "NEXT!".
    Each block is expected to contain an OpenAlex URL like "https://openalex.org/WORKID"
    and a Python-like dictionary (e.g., "{'key': 'value', ...}").

    Returns a DataFrame with columns: openalexid, key, value.
    If output_csv is provided, the DataFrame is saved to that file.

    Notes:
    - Uses ast.literal_eval to safely parse dictionary-like strings.
    - If either the OpenAlex id or a well-formed dictionary cannot be found,
      the corresponding fields are set to "N/A".
    """
    openalexid_list = []
    key_list = []
    value_list = []

    # Read full content
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into chunks using the delimiter
    chunks = content.split("NEXT!")

    for chunk in chunks:
        if not chunk.strip():
            continue  # skip empty chunks

        # 1) Extract OpenAlex id from URL if present
        openalexid_match = re.search(r"https?://openalex\.org/([^\s\n,;]+)", chunk)
        openalexid = openalexid_match.group(1).strip() if openalexid_match else "N/A"

        # 2) Try to find a Python-like dictionary in the chunk.
        #    This pattern looks for the first {...} block (greedy inside braces).
        dict_match = re.search(r"\{.*\}", chunk, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group(0)
            try:
                parsed_obj = ast.literal_eval(dict_str)
                if isinstance(parsed_obj, dict) and parsed_obj:
                    # Take the first key/value pair (ordering as in the dict)
                    first_key = next(iter(parsed_obj.keys()))
                    first_value = parsed_obj.get(first_key, "N/A")
                else:
                    first_key = "N/A"
                    first_value = "N/A"
            except (SyntaxError, ValueError):
                # Parsing failed; return N/A for this chunk
                first_key = "N/A"
                first_value = "N/A"
        else:
            first_key = "N/A"
            first_value = "N/A"

        openalexid_list.append(openalexid)
        key_list.append(first_key)
        value_list.append(first_value)

    df = pd.DataFrame({
        "openalexid": openalexid_list,
        "value": value_list,
        "key": key_list
    })

    if output_csv:
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"Data successfully parsed and saved to {output_csv}")

    return df

# ---- Simple CLI -------------------------------------------------------------
def _print_usage():
    print("Usage: python3 parse_openalex.py <input_file> [output_csv]")
    print("If output_csv is omitted, the function will parse and return the DataFrame only.")

if __name__ == "__main__":  # allow import without running
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        df_result = parse_text_to_csv(input_path, output_path)
        # If no output path was provided, print a small preview
        if output_path is None:
            print(df_result.head())
    except FileNotFoundError:
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)
