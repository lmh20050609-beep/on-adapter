"""Execution gating for the frozen nine-domain V1 campaign.

This package is driven by its own sequential controller
(`benchmark.runner evaluate`), not by the historical parallel
queue. The guard therefore checks only what the controller can actually
guarantee: an explicit authorization flag, an existing unpaused control file,
and a live owner process.
"""
import os
from pathlib import Path

import psutil

from .state_io import read_json

ALLOWED_CAMPAIGNS = ('method_comparison_textrepair_v1_',)


def require_execution(settings, authorized=False, owner_pid=None, job=None, lease_token=None):
    if not authorized or not settings.get('execution', {}).get('gpu_execution_authorized', False):
        raise PermissionError('Execution disabled: explicit authorization and an enabled '
                              'execution configuration are required.')
    if not settings['campaign'].startswith(ALLOWED_CAMPAIGNS):
        raise ValueError('Only the frozen nine-domain V1 campaign is permitted')
    control = Path(settings['execution']['control_file'])
    if not control.exists() or read_json(control).get('pause', True):
        raise PermissionError('Campaign control is paused')
    if owner_pid is not None and not controller_alive(settings, owner_pid):
        raise PermissionError('Controller process is not alive')
    return True


def controller_alive(settings, owner_pid):
    if owner_pid is None:
        return True
    try:
        process = psutil.Process(owner_pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def status(settings=None):
    if settings is None:
        return dict(controller='not_configured')
    pid_file = Path(settings['execution']['report_root']) / 'controller_pid.json'
    record = read_json(pid_file) if pid_file.exists() else {}
    pid = record.get('pid')
    return dict(controller=record, alive=controller_alive(settings, pid) if pid else False)
