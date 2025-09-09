import pandas as pd
import os

def prepare_dpo_dataset(scores_file, pairs_file, resumes_file, output_file,
                        swap_groups=("black", "hispanic"),
                        swap_fraction=1.0,          # set <1.0 to only swap a random subset of matching rows
                        random_state=42,
                        swap_scores=False):
    """
    Creates an updated DPO training dataset with:
    - 'chosen' and 'rejected' columns (renamed from max_value/min_value)
    - 'content' column containing instruction + resume
    - 'score_chosen' and 'score_rejected' from parsed scores
    - OPTIONAL: swap chosen/rejected (and their scores) for rows where resumes_df['race']
      is in swap_groups (e.g., 'black', 'hispanic'), for a configurable fraction.
    """

    # Load input files
    scores_df = pd.read_csv(scores_file)
    pairs_df = pd.read_csv(pairs_file)
    resumes_df = pd.read_csv(resumes_file)

    # Validate required columns
    if 'resume' not in resumes_df.columns:
        raise ValueError("The resumes file must contain a 'resume' column.")
    if 'scores' not in scores_df.columns:
        raise ValueError("The scores file must contain a 'scores' column.")
    if not {'max_value', 'min_value'}.issubset(pairs_df.columns):
        raise ValueError("The pairs file must contain 'max_value' and 'min_value' columns.")

    # Convert scores_df to Series and parse numbers
    scores_series = scores_df.squeeze()
    numbers_expanded = (
        scores_series.str.findall(r'\d+')          # find all numbers as strings
                    .apply(lambda lst: lst[1::2])  # take only value numbers (skip enumeration)
                    .apply(lambda x: list(map(int, x)))  # convert to int
    )

    # Copy and rename columns
    dpo_df = pairs_df.copy()
    dpo_df.rename(columns={'max_value': 'chosen', 'min_value': 'rejected'}, inplace=True)

    # Add content column
    instruction = (
        "You are a resume reviewer.\n\n"
        "Summarize the following resume in 2-3 sentences maximum. "
        "Use the placeholder [NAME] for the person’s name. "
        "Only return the summary, i.e., do not say ‘Here is the summary...’ \n\n"
    )
    dpo_df['content'] = instruction + resumes_df['resume'].astype(str)

    # Add score columns (handle empty lists safely)
    dpo_df['score_chosen'] = numbers_expanded.apply(lambda x: max(x) if x else None)
    dpo_df['score_rejected'] = numbers_expanded.apply(lambda x: min(x) if x else None)

    # ------- NEW: swap chosen/rejected for selected races -------
    # Build a boolean mask for rows whose race is in swap_groups (case-insensitive, robust to NaN)
    race_series = resumes_df.get('race')
    if race_series is not None:
        normalized_race = race_series.astype(str).str.strip().str.lower()
        target_groups = {g.lower() for g in swap_groups}
        candidate_mask = normalized_race.isin(target_groups)

        # Optionally apply only to a random subset of the candidate rows
        if swap_fraction < 1.0 and candidate_mask.any():
            swap_idx = (
                resumes_df[candidate_mask]
                .sample(frac=swap_fraction, random_state=random_state)
                .index
            )
            swap_mask = dpo_df.index.isin(swap_idx)
        else:
            swap_mask = candidate_mask

        # Perform the swap on text columns
        tmp_chosen = dpo_df.loc[swap_mask, 'chosen'].copy()
        dpo_df.loc[swap_mask, 'chosen'] = dpo_df.loc[swap_mask, 'rejected']
        dpo_df.loc[swap_mask, 'rejected'] = tmp_chosen

        # Keep score columns consistent (optional)
        if swap_scores and {'score_chosen', 'score_rejected'}.issubset(dpo_df.columns):
            tmp_score = dpo_df.loc[swap_mask, 'score_chosen'].copy()
            dpo_df.loc[swap_mask, 'score_chosen'] = dpo_df.loc[swap_mask, 'score_rejected']
            dpo_df.loc[swap_mask, 'score_rejected'] = tmp_score
    # ------------------------------------------------------------

    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dpo_df.to_csv(output_file, index=False)
    print(f"✅ Updated DPO training pairs saved to {output_file}")


def main():
    scores_file = "data/generated_scores.csv"
    pairs_file = "data/dpo_training_pairs.csv"
    resumes_file = "data/generated_resumes_with_namesinresume.csv"
    output_file = "data/dpo_training_pairs_updated_swapped.csv"

    prepare_dpo_dataset(scores_file, 
                        pairs_file, 
                        resumes_file, 
                        output_file,
                        swap_groups=("black", "hispanic"),
                        swap_fraction=1.0,      # e.g., 0.5 to swap only half of matching rows
                        random_state=42,
                        swap_scores=False)


if __name__ == "__main__":
    main()
