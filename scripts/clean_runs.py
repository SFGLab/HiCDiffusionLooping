import wandb
from wandb.sdk.wandb_run import Run

if __name__ == '__main__':
    api = wandb.Api()
    project = f"{api.default_entity}/HiCDiffusionLooping"
    runs = api.runs(project)

    to_delete = []
    for run in runs:
        run: Run
        if (
            run.job_type in ('artifact_log', 'datagen') and
            run.state == 'finished' and 
            any(x for x in run.logged_artifacts() if x.type == 'dataset')
        ):
            continue
        print(
            f"Run ID: {run.id}, Name: {run.name}, Status: {run.state}, URL: ", 
            f"https://wandb.ai/{project}/runs/{run.id}"
        )
        to_delete.append(run)

    if not to_delete:
        exit()

    while True:
        inp = input(f'Delete listed ({len(to_delete)}) runs? (yes/no):')
        if inp == 'yes':
            break
        if inp == 'no':
            exit()
    
    for run in to_delete:
        run.delete()
