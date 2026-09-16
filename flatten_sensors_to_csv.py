#!/usr/bin/env python3
"""
Flatten all iotbuilding XMI snapshots in a directory into ONE CSV
(flattened_data.csv), one row per SENSOR, columns = plain sensor attribute
values (as declared in iotbuilding.ecore) plus the snapshot's label.

No encoding, no pooling, no aggregation -- exactly as requested. Room-level
data (roomType, floorArea, etc.) and Connection/topology data are NOT
included; only sensor attributes + label, per the task.

Each snapshot contributes MULTIPLE rows (one per sensor it contains), all
sharing the same snapshot_id and label. This is deliberate: it's the raw
material for a later ML pipeline to decide how to build per-snapshot feature
vectors (e.g. group by snapshot_id and pivot/aggregate) -- that decision is
left for the downstream Random Forest pipeline, not made here.

Schema-driven, not hardcoded: the Sensor attribute list (modality, mean, max,
slope, variance, zScoreVsHistory) is read from the .ecore metamodel itself at
runtime via PyEcore, so this script keeps working if Sensor's attributes ever
change -- no line in this file needs editing for that.

Usage:
    python flatten_sensors_to_csv.py <xmi_dir> [--ecore iotbuilding.ecore]
                                                [--out flattened_data.csv]
"""

import argparse
import csv
import glob
import os

from pyecore.ecore import EAttribute
from pyecore.resources import ResourceSet, URI


def load_metamodel(ecore_path):
    """Load an .ecore file and register its package(s) so instances can be
    resolved against it."""
    rset = ResourceSet()
    resource = rset.get_resource(URI(ecore_path))
    root_package = resource.contents[0]

    def register(pkg):
        rset.metamodel_registry[pkg.nsURI] = pkg
        for sub in getattr(pkg, 'eSubpackages', []):
            register(sub)

    register(root_package)
    return rset, root_package


def load_model(rset, model_path):
    """Load one .xmi instance against the already-registered metamodel."""
    resource = rset.get_resource(URI(model_path))
    return resource.contents[0]


def find_eclass(package, name):
    """Depth-first search for an EClass by name anywhere in the package tree."""
    for c in package.eClassifiers:
        if getattr(c, 'name', None) == name:
            return c
    for sub in getattr(package, 'eSubpackages', []):
        found = find_eclass(sub, name)
        if found is not None:
            return found
    return None


def sensor_attribute_names(root_package):
    """The Sensor EClass's own EAttributes, in declared order. Read from the
    metamodel at runtime -- not hardcoded -- so schema changes need no code
    change here."""
    sensor_cls = find_eclass(root_package, 'Sensor')
    if sensor_cls is None:
        raise RuntimeError(
            "No 'Sensor' EClass found in the metamodel. Check --ecore points "
            "at the right file, or that the class is still named 'Sensor'."
        )
    names = [f.name for f in sensor_cls.eStructuralFeatures if isinstance(f, EAttribute)]
    if not names:
        raise RuntimeError("'Sensor' EClass has no EAttributes -- nothing to extract.")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xmi_dir', help='Directory containing snapshot_*.xmi files')
    parser.add_argument('--ecore', default='iotbuilding.ecore',
                        help='Path to the .ecore metamodel (default: iotbuilding.ecore)')
    parser.add_argument('--out', default='flattened_data.csv',
                        help='Output CSV path (default: flattened_data.csv)')
    args = parser.parse_args()

    rset, root_package = load_metamodel(args.ecore)
    attr_names = sensor_attribute_names(root_package)
    print(f'Sensor attributes (from metamodel, declared order): {attr_names}')

    files = sorted(glob.glob(os.path.join(args.xmi_dir, '*.xmi')))
    print(f'{len(files)} .xmi files found in {args.xmi_dir}')
    if not files:
        raise RuntimeError(f'No .xmi files found in {args.xmi_dir}.')

    rows = []
    n_no_sensors = 0
    for path in files:
        root_obj = load_model(rset, path)
        label = getattr(root_obj, 'label', None)
        # Use the filename stem as snapshot_id -- guaranteed unique within a
        # directory by construction, unlike floorId (which is only unique if
        # the generator happened to keep it so; don't rely on that).
        # floorId is still captured as its own column for reference.
        snapshot_id = os.path.splitext(os.path.basename(path))[0]
        floor_id = getattr(root_obj, 'floorId', None)

        n_sensors_here = 0
        for obj in root_obj.eAllContents():
            if obj.eClass.name != 'Sensor':
                continue
            row = {'snapshot_id': snapshot_id, 'floor_id': floor_id}
            for a in attr_names:
                value = getattr(obj, a, None)
                # Enum literals (e.g. modality) come back as EEnumLiteral --
                # take .name for a plain string, exactly as it appears in the
                # XMI, no encoding.
                row[a] = getattr(value, 'name', value)
            row['label'] = label
            rows.append(row)
            n_sensors_here += 1

        if n_sensors_here == 0:
            n_no_sensors += 1
            print(f'  [warn] no Sensor objects found in {os.path.basename(path)}')

    if not rows:
        raise RuntimeError(
            'No sensor rows extracted from any file -- check the metamodel '
            'and directory are the right ones for these instances.'
        )

    fieldnames = ['snapshot_id', 'floor_id'] + attr_names + ['label']
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nWrote {len(rows)} sensor rows from {len(files)} snapshots '
          f'({n_no_sensors} snapshots had no sensors) to {args.out}')


if __name__ == '__main__':
    main()