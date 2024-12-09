import os
import sys
import datetime as dt
import requests

URL = 'https://api.telegram.org/bot'

def notify(msg):
    try:
        key = os.environ['NOTIFY_BOT_KEY']
        user_id = os.environ.get('NOTIFY_USER_ID')
        if user_id is None:
            response = requests.get(URL + key + '/GetUpdates').json()
            user_id = response['result'][-1]['message']['from']['id']

        text = ''
        if os.environ.get('SLURM_JOB_ID'):
            nodes = os.environ['SLURM_NODELIST']
            name = os.environ['SLURM_JOB_NAME']
            job_id = int(os.environ['SLURM_JOB_ID'])
            start = dt.datetime.fromtimestamp(int(os.environ['SLURM_JOB_START_TIME'])).isoformat(sep=' ')
            end = dt.datetime.fromtimestamp(int(os.environ['SLURM_JOB_END_TIME'])).isoformat(sep=' ')
            now = dt.datetime.now().isoformat(timespec='seconds', sep=' ')
            text = (
                f'```python\n'
                f'  {name=}\n'
                f'{job_id=}\n'
                f' {nodes=}\n'
                f' {start=}\n'
                f'   {now=}\n'
                f'   {end=}\n'
                f'```\n'
            )
        
        requests.post(
            URL + key + "/sendmessage",
            json={"chat_id": user_id, "text": text + msg, 'parse_mode': 'MarkdownV2'},
            timeout=60,
        )

    except Exception as e:
        print(type(e), str(e))


if __name__ == '__main__':
    notify(sys.argv[1])
