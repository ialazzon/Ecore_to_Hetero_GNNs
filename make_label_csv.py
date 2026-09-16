#!/usr/bin/env python3
"""
Build label_of.csv from the snapshot XMI files in xmi_new/.

For each snapshot_ddddd.xmi, read the graph-classification label from the
BuildingSnapshot root object's 'label' attribute and write one CSV row:

    snapshot_00001,HVACFault

The label is also printed to stdout as it is read.
"""

import os
import glob
import csv
from lxml import etree

IN_DIR = "xmi_new"
OUT_CSV = "label_of.csv"


def read_label(xmi_path):
    """Return the 'label' attribute of the BuildingSnapshot root element."""
    root = etree.parse(xmi_path).getroot()
    # root is the BuildingSnapshot; the label lives directly on it as an attr
    return root.get("label")


def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "snapshot_*.xmi")))
    if not files:
        print(f"No snapshot_*.xmi files found in {IN_DIR}/")
        return

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        for path in files:
            name = os.path.splitext(os.path.basename(path))[0]  # snapshot_00001
            label = read_label(path)
            print(f"{name},{label}")
            writer.writerow([name, label])

    print(f"\nWrote {len(files)} labels to {OUT_CSV}")


if __name__ == "__main__":
    main()