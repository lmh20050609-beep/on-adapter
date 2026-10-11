"""OS state is authoritative; suspended processes still own their execution lock."""
import json
import os
import time
from pathlib import Path
import psutil
from .config import ROOT
from .state_io import read_json


def classify_process(process, suspension_record=None):
    raw=process.status()
    if raw==psutil.STATUS_STOPPED: return {'state':'os_suspended','raw_status':raw}
    # Windows may create an idle driver helper thread after NtSuspendProcess.
    # psutil then says "running" although the Python thread is still suspended.
    # Do not suspend again (that would alter suspend counts), or infer activity
    # from that string alone. Confirm identity, CPU totals and thread wait state.
    r=suspension_record
    if r and os.name=='nt' and process.create_time()==r['create_time'] and process.cmdline()==r['command']:
        import subprocess
        command=("(Get-Process -Id %d).Threads | Select-Object Id,@{N='State';E={$_.ThreadState.ToString()}}," 
                 "@{N='Wait';E={if($_.ThreadState -eq 'Wait'){$_.WaitReason.ToString()}else{'not_waiting'}}},"
                 "@{N='CPU';E={$_.TotalProcessorTime.TotalSeconds}} | ConvertTo-Json -Compress")%process.pid
        result=subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,
                              creationflags=subprocess.CREATE_NO_WINDOW,timeout=20,check=True)
        threads=json.loads(result.stdout);threads=[threads] if isinstance(threads,dict) else threads
        suspended=[t for t in threads if t['State']=='Wait' and t['Wait']=='Suspended']
        others=[t for t in threads if t not in suspended]
        if suspended and all(t['State']=='Wait' and t['CPU']==0 for t in others):
            return dict(state='os_suspended_with_idle_auxiliary_threads',raw_status=raw,
                        suspended_threads=len(suspended),idle_auxiliary_threads=others,
                        suspended_thread_cpu={str(t['Id']):t['CPU'] for t in suspended},
                        aggregate_cpu_delta_since_pause=sum(process.cpu_times()[:2])-r['cpu_seconds'],
                        note='Aggregate process CPU may include transient OS/driver helper threads; original execution threads are suspended')
    return {'state':'running_or_unverified','raw_status':raw}


def process_inventory():
    results=[]
    path=ROOT/'reports/method_comparison_20261004/suspended_processes.json'
    records={r['pid']:r for r in json.loads(path.read_text('utf-8'))['processes']} if path.exists() else {}
    for p in psutil.process_iter(['pid','ppid','create_time','cmdline','status']):
        try:
            cmd=p.info['cmdline'] or []
            if not any(x in cmd for x in ['method_comparison.queue','method_comparison.worker',
                                          'method_comparison_optimized.queue','method_comparison_optimized.worker',
                                          'method_comparison_optimized_v2.queue','method_comparison_optimized_v2.worker',
                                          'method_comparison_optimized_v3.queue','method_comparison_optimized_v3.worker',
                                          'method_comparison_optimized_v4.queue','method_comparison_optimized_v4.worker',
                                          'method_comparison_periodic_v1.queue','method_comparison_periodic_v1.worker',
                                          'method_comparison_periodic_v2.queue','method_comparison_periodic_v2.worker',
                                          'method_comparison_periodic_v3.queue','method_comparison_periodic_v3.worker',
                                          'method_comparison_periodic_v4.queue','method_comparison_periodic_v4.worker']): continue
            if (any(x in cmd for x in ['method_comparison_optimized.queue','method_comparison_optimized_v2.queue','method_comparison_optimized_v3.queue','method_comparison_optimized_v4.queue','method_comparison_periodic_v1.queue','method_comparison_periodic_v2.queue','method_comparison_periodic_v3.queue','method_comparison_periodic_v4.queue'])
                    and '--execute-authorized' not in cmd):
                continue  # Read-only plans own no execution slot or lock.
            observed=classify_process(p,records.get(p.pid))
            results.append(dict(p.info,execution_state=observed['state'],observation=observed))
        except (psutil.NoSuchProcess,psutil.AccessDenied): continue
    return results


def status():
    state_path=ROOT/'reports/method_comparison_20261004/queue_state.json'
    state=json.loads(state_path.read_text('utf-8'))
    live=process_inventory()
    record_file=state_path.parent/'suspended_processes.json'
    records={r['pid']:r for r in json.loads(record_file.read_text('utf-8'))['processes']} if record_file.exists() else {}
    expected=records.get(state.get('pid'))
    matching=[p for p in live if p['pid']==state.get('pid') and 'method_comparison.queue' in p['cmdline']
              and (expected is None or p['create_time']==expected['create_time'])]
    optimized_path=ROOT/'reports/periodic_guard_repair_20261007/queue_state.json'
    optimized=json.loads(optimized_path.read_text('utf-8')) if optimized_path.exists() else None
    optimized_live=[p for p in live if optimized and p['pid']==optimized.get('pid')
                    and 'method_comparison_periodic_v4.queue' in p['cmdline']
                    and ('create_time' not in optimized or p['create_time']==optimized['create_time'])]
    return dict(processes=live,legacy_state=state,
                legacy_actual=matching[0]['execution_state'] if matching else 'exited_or_pid_reused',
                stale_state=not matching and state.get('status') in ['running','paused_suspended_by_user'],
                pause=json.loads((state_path.parent/'control.json').read_text('utf-8'))['pause'],
                optimized_state=optimized,optimized_actual=optimized_live[0]['execution_state'] if optimized_live else
                    ('not_started' if optimized is None else 'exited_or_pid_reused'),
                optimized_stale_state=bool(optimized and optimized.get('status')=='running' and not optimized_live))


def refresh_snapshot(snapshot):
    """None means a proven exit; access errors and PID reuse are not ignored."""
    try:
        process = psutil.Process(snapshot['pid'])
        if process.create_time() != snapshot['create_time']:
            raise RuntimeError('PID reused while validating training process')
        value = dict(snapshot, ppid=process.ppid(), cmdline=process.cmdline(), status=process.status())
        parents = {p.pid for p in process.parents()}
        if not process.is_running():
            return None
        return value, parents
    except psutil.NoSuchProcess:
        return None


def require_execution(settings, authorized=False, owner_pid=None, job=None, lease_token=None):
    if not authorized or not settings.get('execution',{}).get('gpu_execution_authorized',False):
        raise PermissionError('Execution disabled: this delivery authorizes CPU validation only. Explicit user authorization and an enabled execution configuration are required.')
    if not settings['campaign'].startswith('method_comparison_opt_'):
        raise ValueError('Only an isolated optimized campaign is permitted')
    control=Path(settings['execution']['control_file'])
    if not control.exists() or read_json(control).get('pause',True):
        raise PermissionError('Optimized control is paused')
    allowed={os.getpid()}
    # Windows venv redirectors have the same module arguments as their child.
    parent=psutil.Process().parent()
    if parent: allowed.add(parent.pid)
    if owner_pid is not None:
        owner=psutil.Process(owner_pid)
        if 'method_comparison_periodic_v4.queue' not in owner.cmdline(): raise ValueError('Invalid queue owner')
        if owner.pid not in {p.pid for p in psutil.Process().parents()}: raise ValueError('Worker is not a descendant of its queue')
        allowed.add(owner.pid)
        if owner.parent(): allowed.add(owner.parent().pid)
        from .resources import identity, digest
        from .parallel_policy import command_matches_lease, job_key
        registry=Path(settings['execution']['report_root'])/'active_workers.json'
        session=read_json(registry)
        if (session['pid']!=owner_pid or session['create_time']!=owner.create_time()
                or session['settings_identity']!=identity(settings)
                or digest(session['config_path'])!=session['config_sha256']):
            raise ValueError('Controller identity/configuration changed')
        def validated_leases(value):
            if (value['pid']!=owner_pid or value['create_time']!=owner.create_time()
                    or value['settings_identity']!=identity(settings)
                    or value['config_path']!=session['config_path']
                    or value['config_sha256']!=session['config_sha256']
                    or value['session_token']!=session['session_token']):
                raise ValueError('Controller session identity changed during startup')
            leases=value['active']
            matching=[entry for entry in leases if entry['backend']=='periodic' and entry['token']==lease_token]
            if len(matching)!=1 or job is None or job_key(matching[0]['job'])!=job_key(job):
                raise ValueError('Missing or incompatible configuration lease')
            if len({job_key(entry['job']) for entry in leases})!=len(leases):
                raise ValueError('Controller registered duplicate configurations')
            if len(leases)>settings['execution'].get('parallel_workers',1):
                raise ValueError('Controller exceeded configured concurrency')
            return leases
        leases=validated_leases(session)
        # Inspect one process snapshot. A legitimate sibling may be admitted
        # after our first registry read; refresh its lease from the same bound
        # session. Do not compare a second, newer process list to an old set.
        for p in process_inventory():
            if p['pid'] in allowed: continue
            try:
                process=psutil.Process(p['pid'])
                parents={a.pid for a in process.parents()}
            except psutil.NoSuchProcess:
                continue
            if not any(command_matches_lease(p,entry,owner_pid,parents) for entry in leases):
                leases=validated_leases(read_json(registry))
                if not any(command_matches_lease(p,entry,owner_pid,parents) for entry in leases):
                    # A sibling can finish between the inventory snapshot and
                    # the refreshed registry. Reobserve its exact PID identity
                    # before declaring a live, unregistered training process.
                    fresh = refresh_snapshot(p)
                    if fresh is None:
                        continue
                    p, parents = fresh
                    leases = validated_leases(read_json(registry))
                    if any(command_matches_lease(p,entry,owner_pid,parents) for entry in leases):
                        allowed.add(p['pid'])
                        continue
                    # A second narrow check closes exit during the registry read.
                    if refresh_snapshot(p) is None:
                        continue
                    raise RuntimeError('Unregistered queue/worker, including suspended tasks: '+json.dumps(p))
            allowed.add(p['pid'])
        return  # All observed siblings were validated against current leases.

    conflicts=[p for p in process_inventory() if p['pid'] not in allowed]
    if conflicts: raise RuntimeError('Another queue/worker exists, including OS-suspended tasks: '+json.dumps(conflicts))


def controller_alive(settings, owner_pid):
    """Cheap checkpoint-boundary check; PID reuse is never accepted."""
    if owner_pid is None: return True
    try:
        session=read_json(Path(settings['execution']['report_root'])/'active_workers.json')
        owner=psutil.Process(owner_pid)
        from .resources import identity
        return (owner.is_running() and owner.create_time()==session['create_time']
                and session['pid']==owner_pid and session['settings_identity']==identity(settings)
                and 'method_comparison_periodic_v4.queue' in owner.cmdline())
    except (psutil.NoSuchProcess,FileNotFoundError):
        return False


def main():
    print(json.dumps(status(),indent=2,default=str))


if __name__=='__main__':main()
