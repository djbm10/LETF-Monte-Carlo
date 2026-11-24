from pathlib import Path
import pytest
import yaml
from scripts.check_workflows import UniqueKeyLoader, check


def test_duplicate_jobs_rejected():
    with pytest.raises(ValueError, match='Duplicate YAML key'):
        yaml.load('jobs:\n  research: {}\n  research: {}\n', Loader=UniqueKeyLoader)


def test_every_repository_workflow_has_unique_keys():
    assert check(Path(__file__).resolve().parents[1]/'.github/workflows') > 0
