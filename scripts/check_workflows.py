"""Reject duplicate YAML mapping keys before GitHub silently has no runnable jobs."""
from pathlib import Path
import yaml


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f'Duplicate YAML key {key!r}, line {key_node.start_mark.line + 1}')
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def check(folder=Path('.github/workflows')):
    paths = sorted(list(folder.glob('*.yml')) + list(folder.glob('*.yaml')))
    for path in paths:
        try:
            workflow = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
            if not isinstance(workflow, dict) or not isinstance(workflow.get('jobs'), dict):
                raise ValueError('Workflow requires a jobs mapping')
        except Exception as exc:
            raise ValueError(f'{path}: {exc}') from exc
    return len(paths)


if __name__ == '__main__':
    print(f'{check()} workflows: unique YAML mappings and jobs structure PASS')
