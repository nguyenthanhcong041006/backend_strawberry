from pathlib import Path
import pandas as pd
import json


def create_eol_anchors(output_csv: Path, active_dataset: str):

    data = []
    for i in range (1, 7):
            
        data.append(
        {
            "experiment_id": f"{active_dataset}_experiment",
            "fruit_id": f"F{i:02d}",
            "eol_timestamp": "2026-03-26 08:00:00", # CHANGE THE END OF LIFE TIMESTAMP HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            "eol_basis": "visual",
            "proposed_by": "cong",
            "reviewed_by": "",
            "approved_by": "",
            "status": "approved",
            "notes": "End of life"
        }
            )
    

    df = pd.DataFrame(data)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print(f"Saved: {output_csv}")
    print(df)


def main():

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    
    CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        configs = json.load(f)

    active_dataset = configs["active_dataset"]

    output_csv = (
        PROJECT_ROOT
        / "data"
        / "02_processed"
        / active_dataset
        / "manifests"
        / "eol_anchors.csv"
    )

    create_eol_anchors(output_csv, active_dataset)


if __name__ == "__main__":
    main()