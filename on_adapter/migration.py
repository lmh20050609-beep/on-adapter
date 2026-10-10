"""Fresh three-seed campaign: external recovery forbidden; own ResumeFile is unchanged."""
def continuation_jobs(settings, known, approved):
    from .config import configured_windows
    configured_windows(settings)
    return set()

def lineage(folder):
    if (folder/'migration.json').exists():
        raise ValueError('Prior-campaign migration is forbidden in the fresh three-seed campaign')
    return None

def load_migrated(folder, metadata):
    lineage(folder)
    return None
