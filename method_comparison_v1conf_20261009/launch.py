"""Explicitly authorized Windows WMI launch. Never resumes/kills legacy tasks."""
import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from .config import ROOT,load_config
from .process_guard import require_execution
from .resources import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute-authorized',action='store_true')
    parser.add_argument('--config',default=str(ROOT/'configs/method_comparison_periodic_v3_20261006.json'))
    args=parser.parse_args();settings=load_config(args.config)
    if not args.execute_authorized:
        print('Plan only. Use python -m method_comparison_periodic_v3.queue to inspect the matrix. No process started.');return
    require_execution(settings,True)
    report=Path(settings['execution']['report_root']);report.mkdir(parents=True,exist_ok=True)
    if (report/'queue.lock').exists(): raise RuntimeError('Queue lock exists; inspect owner before doing anything')
    stamp=time.strftime('%Y%m%d_%H%M%S');log=report/('queue_'+stamp+'.log')
    command=subprocess.list2cmdline([str(ROOT/'.venv/Scripts/pythonw.exe'),'-m','method_comparison_periodic_v3.background',
        '--log',str(log),'-m','method_comparison_periodic_v3.queue','--config',str(Path(args.config).resolve()),'--execute-authorized'])
    payload=base64.b64encode(json.dumps(dict(command=command,cwd=str(ROOT))).encode('utf-8')).decode('ascii')
    script=("$p=([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('%s')))|ConvertFrom-Json;"
            "$s=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0};"
            "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$p.command;"
            "CurrentDirectory=$p.cwd;ProcessStartupInformation=$s}|ConvertTo-Json -Compress")%payload
    encoded=base64.b64encode(script.encode('utf-16le')).decode('ascii')
    result=subprocess.run(['powershell.exe','-NoProfile','-EncodedCommand',encoded],capture_output=True,text=True,
                          encoding='utf-8',errors='replace',
                          creationflags=subprocess.CREATE_NO_WINDOW,check=True,timeout=30)
    record=json.loads(result.stdout)
    if record['ReturnValue']!=0:raise RuntimeError('WMI rejected process creation: '+result.stdout)
    atomic_json(report/('launch_'+stamp+'.json'),dict(wmi=record,command=command,log=str(log),created_at=time.time()))
    print(json.dumps(record))


if __name__=='__main__':main()
