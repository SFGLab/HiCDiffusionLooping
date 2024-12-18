import wandb
import pandas as pd

if __name__ == '__main__':
    run = wandb.init(project="HiCDiffusionLooping", job_type="artifact_log")

    def _read_supp_files(path):
        supp_files = pd.read_csv(path, sep='\t', skiprows=1)
        supp_files = supp_files[~supp_files["File Download URL"].str.contains('#')]
        return supp_files
    
    supp_files = [
        dict(prefix='wtc11_rep1_', subdir='wtc11/', path='data/4DNESHWX9JLY_wtc11_biorep1.tsv'),
        dict(prefix='wtc11_rep2_', subdir='wtc11/', path='data/4DNESHWX9JLY_wtc11_biorep2.tsv')
    ]

    for supp_file in supp_files:
        for x in _read_supp_files(supp_file['path']).to_dict('records'):
            artifact_4dn = wandb.Artifact(
                name=supp_file['prefix'] + x['File Accession'],
                type="dataset",
                description=f"{x['File Format']}: {x['File Type']}, {x['Replicate Info']}, {x['Biosource']}",
                metadata=x,
            )
            artifact_4dn.add_reference(
                uri=x["Open Data URL"],
                name=supp_file['subdir'] + supp_file['prefix'] + x["Open Data URL"].split('/')[-1]
            )
            print(run.log_artifact(artifact_4dn))

    run.finish()
