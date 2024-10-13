import wandb
import pandas as pd

if __name__ == '__main__':

    run = wandb.init(project="HiCDiffusionLooping", job_type="artifact_log")

    artifact = wandb.Artifact(
        name="GRCh38-reference-genome",
        type="dataset",
        description="GRCh38_full_analysis_set_plus_decoy_hla.fa from 1000genomes"
    )

    artifact.add_reference(
        "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa"
    )
    run.log_artifact(artifact)

    supp_files = pd.read_csv('data/4DNES7IB5LY9_biorep2.tsv',
                             sep='\t',
                             skiprows=1)
    supp_files = supp_files[~supp_files["File Download URL"].str.contains('#')]

    for x in supp_files.to_dict('records'):
        artifact_4dn = wandb.Artifact(
            name=x['File Accession'],
            type="dataset",
            description=
            f"{x['File Format']}: {x['File Type']}, {x['Replicate Info']}",
            metadata=x,
        )
        artifact_4dn.add_reference(x["Open Data URL"])
        print(run.log_artifact(artifact_4dn))

    
    small_files = pd.read_csv('data/4DN_small_files.tsv', sep='\t')
    for x in small_files.to_dict('records'):
        artifact_4dn = wandb.Artifact(
            name=x['id'],
            type="dataset",
            description=x['desc'],
            metadata=x,
        )
        artifact_4dn.add_reference(x["url"])
        print(run.log_artifact(artifact_4dn))

    for motif_url, desc in [
        ('https://jaspar.elixir.no/api/v1/matrix/MA1929.2.meme', 'TF has several motif variants. Extended motif with zinc finger 8 (5bp)'),
        ('https://jaspar.elixir.no/api/v1/matrix/MA1930.2.meme', 'TF has several motif variants. Extended motif with zinc finger 8 (6bp)'),
        ('https://jaspar.elixir.no/api/v1/matrix/MA0139.2.meme', 'TF has several motif variants.'),
    ]:
        artifact_motif = wandb.Artifact(
            name=motif_url.split('/')[-1],
            type="dataset",
            description=desc
        )
        artifact_motif.add_reference(motif_url)
        print(run.log_artifact(artifact_motif))

    run.finish()
