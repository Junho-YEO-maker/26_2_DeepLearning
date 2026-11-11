"""Read the detection team's shared, repository-relative input lists."""
import csv
from pathlib import Path, PurePosixPath

PCB_TYPES = ('pcb1', 'pcb2', 'pcb3')
DEFECT_TYPES = {1: 'Bent', 2: 'Melt', 3: 'Scratch', 4: 'Missing'}
SPLITS = Path('Dataset/visa/splits')
RAW = PurePosixPath('Dataset/visa/raw')


def raw_path(repo_root, row, key):
    value = row[key]
    path = PurePosixPath(value)
    parent = RAW / row['category'] / 'Data'
    folder = parent / 'Masks' / 'Anomaly' if key == 'mask' else (
        parent / 'Images' / ('Normal' if row['label'] == '0' else 'Anomaly')
    )
    if (path.is_absolute() or '..' in path.parts or '\\' in value or str(path) != value
            or path.parent != folder):
        raise ValueError(f'Expected a repository-relative VisA path: {value!r}')
    return repo_root / path


def read_split(repo_root, filename, expected_label=None):
    path = repo_root / SPLITS / filename
    if not path.is_file():
        raise FileNotFoundError(f'Shared split CSV is missing: {SPLITS / filename}')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'category', 'image', 'mask', 'label'}
        if expected_label == '1':
            required.add('defect_type')
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'{filename}: missing columns {sorted(required - set(reader.fieldnames or ()))}')
        rows = list(reader)

    images = set()
    for row in rows:
        if row['category'] not in PCB_TYPES or row['label'] not in {'0', '1'}:
            raise ValueError(f'{filename}: invalid category or label for {row["image"]!r}')
        if expected_label is not None and row['label'] != expected_label:
            raise ValueError(f'{filename}: unexpected label for {row["image"]!r}')
        raw_path(repo_root, row, 'image')
        if row['label'] == '1':
            raw_path(repo_root, row, 'mask')
        elif row['mask']:
            raise ValueError(f'{filename}: normal image has a mask')
        if row['image'] in images:
            raise ValueError(f'{filename}: duplicate image {row["image"]}')
        images.add(row['image'])
    return rows


def load_team_split(repo_root, k, seed):
    """Return allowed normal rows and single-type defect rows for the main experiment."""
    if k < 1 or seed < 0:
        raise ValueError('Generation requires K >= 1 and seed >= 0')
    repo_root = Path(repo_root)
    test = read_split(repo_root, 'test.csv')
    validation = read_split(repo_root, 'val_normal.csv', '0')
    normals = read_split(repo_root, 'train_normal.csv', '0')
    defects = read_split(repo_root, f'train_K{k}_seed{seed}.csv', '1')
    if not test or not validation:
        raise ValueError('Shared test and validation splits must contain rows')
    # Compare resolved paths too, so a Drive symlink cannot create an alias.
    seen = set()
    for rows in (test, validation, normals, defects):
        current = {raw_path(repo_root, row, 'image').resolve() for row in rows}
        if len(current) != len(rows):
            raise ValueError('Duplicate resolved image within a shared split')
        overlap = seen & current
        if overlap:
            raise ValueError(f'Overlapping shared split image: {min(overlap)}')
        seen.update(current)

    names = {name.casefold(): class_id for class_id, name in DEFECT_TYPES.items()}
    grouped_normal = {pcb: [] for pcb in PCB_TYPES}
    grouped_defect = {(pcb, class_id): [] for pcb in PCB_TYPES for class_id in DEFECT_TYPES}
    for row in normals:
        grouped_normal[row['category']].append(row)
    for row in defects:
        name = (row['defect_type'] or '').strip().casefold()
        if name not in names:
            raise ValueError(f'Expected one named defect type for {row["image"]!r}')
        grouped_defect[row['category'], names[name]].append(row)
    for pcb, group in grouped_normal.items():
        if not group:
            raise ValueError(f'No normal training images for {pcb}')
    for (pcb, class_id), group in grouped_defect.items():
        if len(group) != k:
            raise ValueError(f'{pcb} {DEFECT_TYPES[class_id]}: expected K={k}, found {len(group)}')
    return grouped_normal, grouped_defect
