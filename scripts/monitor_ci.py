#!/usr/bin/env python3
import time, urllib.request, json, sys

url = 'https://api.github.com/repos/dilshanrathnaweera/demo_Smart_ClassRoom/actions/runs?branch=ci/docker-test'
print('Monitoring CI for branch ci/docker-test...')
max_checks = 120
for i in range(max_checks):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        print('ERROR fetching runs:', e)
        time.sleep(5)
        continue
    runs = data.get('workflow_runs', [])
    if not runs:
        print('No workflow runs found yet.')
        time.sleep(5)
        continue
    run = runs[0]
    run_id = run.get('id')
    status = run.get('status')
    conclusion = run.get('conclusion')
    html_url = run.get('html_url')
    head_sha = run.get('head_sha')
    print(f"Run id={run_id} status={status} conclusion={conclusion} url={html_url}")
    if status != 'completed':
        time.sleep(5)
        continue
    print('\nWorkflow finished:')
    print('  id:', run_id)
    print('  url:', html_url)
    print('  status:', status)
    print('  conclusion:', conclusion)
    jobs_url = f"https://api.github.com/repos/dilshanrathnaweera/demo_Smart_ClassRoom/actions/runs/{run_id}/jobs"
    try:
        with urllib.request.urlopen(jobs_url, timeout=30) as r:
            jobs_data = json.load(r)
    except Exception as e:
        print('ERROR fetching jobs:', e)
        sys.exit(0)
    jobs = jobs_data.get('jobs', [])
    for j in jobs:
        print('\nJob:', j.get('name'))
        print('  id:', j.get('id'))
        print('  status:', j.get('status'))
        print('  conclusion:', j.get('conclusion'))
        steps = j.get('steps', [])
        for s in steps:
            print('   - step:', s.get('name'), 'status:', s.get('status'), 'conclusion:', s.get('conclusion'))
            if s.get('conclusion') == 'failure':
                print('     failure details:', s.get('number'), s.get('completed_at'))
    if conclusion != 'success':
        print('\nCI failed. Suggestions:')
        print('- Open the workflow run URL to view logs: ' + html_url)
        print('- Check failing job steps printed above; inspect step logs for stack traces and missing dependencies.')
    else:
        print('\nCI succeeded.')
    sys.exit(0)
print('Timeout waiting for workflow to complete')
