"""
Convert an Ecore metamodel (.ecore) and a conforming model instance (.model)
into a PyTorch Geometric HeteroData graph.

Design rules:
  1. The root object is ignored (no node for it).
  2. Node types = EClasses that the ROOT class directly composes (i.e.
     EReferences on the root EClass with containment=True). Classes nested
     deeper in the containment tree are NOT automatically node types.
  3. Association classes: an EClass AC qualifies as an association class
     joining two node types if:
       (a) AC is composed (containment=True) by exactly one owner class,
           and that owner is itself a node type, AND
       (b) AC declares exactly one outgoing reference to another class,
           and that target is itself a node type.
     Such classes are NOT instantiated as nodes. Each AC instance becomes
     a single edge (owner -> target) whose edge type is named after AC,
     and whose edge_attr is built from AC's own EAttributes.
  4. All other references between two node-type objects (containment or
     not, e.g. self-loops like "follows", or plain refs like "develops")
     become edges with NO features (edge_index only).

Example (matches the GameInstance metamodel):
  - Node types: User, Game, Dev  (direct composition children of root)
  - Plays: composed under User, with one outgoing ref "game" -> Game
      => association class => edge type (User, "Plays", Game), edge_attr = [hours]
  - User.follows (User -> User, plain ref) => edge type (User, "follows", User), no features
  - Dev.develops (Dev -> Game, plain ref)   => edge type (Dev, "develops", Game), no features
"""

import argparse
import hashlib
import os
import glob

import torch
from torch_geometric.data import HeteroData

from pyecore.resources import ResourceSet, URI
from pyecore.ecore import EObject, EClass, EEnum, EEnumLiteral


# --------------------------------------------------------------------------- #
# Value / attribute encoding helpers
# --------------------------------------------------------------------------- #

def _stable_hash(s: str, mod: int = 10 ** 6) -> int:
    """Deterministic string -> int hash (stable across runs, unlike hash())."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


def _is_enum_attr(attr):
    """True if this EAttribute is typed by an EEnum (a fixed set of literals)."""
    etype = getattr(attr, "eType", None)
    return isinstance(etype, EEnum)


def _enum_literal_names(attr):
    """Ordered list of literal names for an enum-typed attribute, taken from the
    METAMODEL (not the data), so the one-hot columns are identical for every
    graph regardless of which literals happen to appear. Order follows the
    enum's declared eLiterals order for stability across runs and datasets."""
    etype = attr.eType
    return [lit.name for lit in etype.eLiterals]


def encode_scalar(value) -> float:
    """Encode a single non-enum, non-list attribute value into one float."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, EEnumLiteral):
        # Should be handled via one-hot upstream; fall back to the integer code
        # rather than a hash if we ever reach here.
        return float(value.value)
    if isinstance(value, EObject):
        name = getattr(value, "name", None)
        return float(_stable_hash(name if name is not None else str(value)))
    return float(_stable_hash(str(value)))


def _enum_value_name(value):
    """Get the literal name from an enum attribute value (or None)."""
    if value is None:
        return None
    if isinstance(value, EEnumLiteral):
        return value.name
    # pyecore sometimes returns the literal directly; guard for str too
    return getattr(value, "name", str(value))


# --------------------------------------------------------------------------- #
# Column schema: decide, per attribute, how many columns it occupies and the
# header names. Enums expand to one column per literal (one-hot); everything
# else is a single scalar column. Driven entirely by the metamodel so the
# schema is identical across all graphs in a dataset.
# --------------------------------------------------------------------------- #

def build_column_schema(attrs_in_order):
    """
    Given an ordered list of EAttributes, return:
      columns : list of (attr, kind, extra) describing each output column group
                kind in {"scalar","enum"}; for "enum", extra = [literal names]
      headers : flat list of column header strings (for debugging/inspection)
    """
    columns = []
    headers = []
    for attr in attrs_in_order:
        if _is_enum_attr(attr) and not attr.many:
            lits = _enum_literal_names(attr)
            columns.append((attr, "enum", lits))
            headers.extend(f"{attr.name}={lit}" for lit in lits)
        else:
            columns.append((attr, "scalar", None))
            headers.append(attr.name)
    return columns, headers


def encode_row(obj, columns):
    """Encode one object into a flat feature row per the column schema."""
    obj_attrs = {a.name: a for a in obj.eClass.eAllAttributes()}
    row = []
    for attr, kind, extra in columns:
        present = attr.name in obj_attrs
        if kind == "enum":
            lits = extra
            if not present:
                row.extend(0.0 for _ in lits)
                continue
            try:
                value = getattr(obj, attr.name)
            except Exception:
                value = None
            name = _enum_value_name(value)
            # one-hot: 1.0 in the matching literal column, 0.0 elsewhere.
            # unset/None enum -> all zeros (a valid "no category" encoding).
            row.extend(1.0 if name == lit else 0.0 for lit in lits)
        else:  # scalar
            if not present:
                row.append(0.0)
                continue
            a = obj_attrs[attr.name]
            if a.many:
                try:
                    value = getattr(obj, a.name)
                    row.append(float(len(value)) if value is not None else 0.0)
                except (TypeError, Exception):
                    row.append(0.0)
            else:
                try:
                    row.append(encode_scalar(getattr(obj, a.name)))
                except Exception:
                    row.append(0.0)
    return row


def build_feature_matrix(objs):
    """
    Build (headers, feature_tensor) for a list of EObjects.

    Enum-typed attributes are ONE-HOT encoded using the full literal set from
    the metamodel, so the produced columns are stable dataset-wide (every graph
    gets the same columns in the same order, whether or not a given literal
    appears in that graph). Non-enum attributes remain single scalar columns.
    """
    # Ordered union of attributes declared across the objects. We SORT by name
    # so the column schema is deterministic across loads/graphs — pyecore's
    # eAllAttributes() iteration order is NOT stable across reloads, which would
    # otherwise scramble feature columns between graphs in the same dataset.
    # ID attributes (iD=True) are identifiers, not features, and are excluded.
    attrs_by_name = {}
    for obj in objs:
        for attr in obj.eClass.eAllAttributes():
            if getattr(attr, "iD", False):
                continue  # skip identifier attributes (e.g. Room.id)
            attrs_by_name.setdefault(attr.name, attr)
    attrs_in_order = [attrs_by_name[name] for name in sorted(attrs_by_name)]

    columns, headers = build_column_schema(attrs_in_order)

    rows = [encode_row(obj, columns) for obj in objs]

    if headers:
        return headers, torch.tensor(rows, dtype=torch.float)
    return headers, torch.zeros((len(objs), 0), dtype=torch.float)


# --------------------------------------------------------------------------- #
# Loading metamodel / model with PyEcore
# --------------------------------------------------------------------------- #

def load_metamodel(ecore_path: str):
    rset = ResourceSet()
    resource = rset.get_resource(URI(ecore_path))
    root_package = resource.contents[0]

    def register(pkg):
        rset.metamodel_registry[pkg.nsURI] = pkg
        for sub in getattr(pkg, "eSubpackages", []):
            register(sub)

    register(root_package)
    return rset, root_package


def load_model(rset: ResourceSet, model_path: str):
    resource = rset.get_resource(URI(model_path))
    return resource.contents[0]


# --------------------------------------------------------------------------- #
# Metamodel analysis: node types vs. association classes
# --------------------------------------------------------------------------- #

def collect_eclasses(root_package):
    """Recursively gather all EClass classifiers from a package tree."""
    classes = []

    def walk(pkg):
        for classifier in pkg.eClassifiers:
            if isinstance(classifier, EClass):
                classes.append(classifier)
        for sub in getattr(pkg, "eSubpackages", []):
            walk(sub)

    walk(root_package)
    return classes


def determine_node_type_classes(root_obj):
    """
    Node types = EClasses directly composed by the ROOT class, i.e. the
    eType of every containment EReference declared on root_obj.eClass.
    """
    root_class = root_obj.eClass
    node_classes = set()
    for ref in root_class.eAllReferences():
        if ref.containment:
            node_classes.add(ref.eType)
    return node_classes


def classify_association_classes(eclasses, node_type_classes):
    """
    Identify EClasses that act as "association classes" joining two node
    types: composed by exactly one owner class (which must itself be a
    node type), and declaring exactly one outgoing reference to another
    class (which must also be a node type).

    Returns: dict {EClass: {
        "owner_class": EClass, "owner_ref": EReference,
        "target_class": EClass, "target_ref": EReference,
    }}
    """
    # Map: EClass -> list of (owner_class, containment_ref) pointing to it
    incoming_containment = {}
    for c in eclasses:
        for ref in c.eAllReferences():
            if ref.containment:
                target_cls = ref.eType
                incoming_containment.setdefault(target_cls, []).append((c, ref))

    association_info = {}
    for c in eclasses:
        if c in node_type_classes:
            continue  # node types themselves are never association classes

        owners = incoming_containment.get(c, [])
        if len(owners) != 1:
            continue
        owner_class, owner_ref = owners[0]
        if owner_class not in node_type_classes:
            continue  # must be composed under a node type

        outgoing = [r for r in c.eAllReferences() if r.eType is not c]
        if len(outgoing) != 1:
            continue
        target_ref = outgoing[0]
        target_class = target_ref.eType
        if target_class not in node_type_classes:
            continue  # must link to another node type

        association_info[c] = {
            "owner_class": owner_class,
            "owner_ref": owner_ref,
            "target_class": target_class,
            "target_ref": target_ref,
        }

    return association_info


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def build_hetero_graph(root_obj, root_package) -> HeteroData:
    """Build a HeteroData graph from an EMF model instance.

    Directionality rule (governs every edge):
      * A reference that declares an eOpposite is BIDIRECTIONAL in the
        metamodel, so it becomes a bidirectional edge: the forward relation
        PLUS a synthesized reverse relation 'rev_<name>' that reuses the
        forward edge_attr. The eOpposite PARTNER reference is suppressed so the
        same logical link is not emitted twice.
      * A reference with NO eOpposite is UNIDIRECTIONAL and becomes a single
        directed edge with no synthesized reverse.
    This holds for both association-class (featured) edges and plain edges.
    Example: Connection.room has eOpposite room_connection -> bidirectional;
             Plays.game / User.follows / Dev.develops have none -> directed.
    """
    data = HeteroData()

    eclasses = collect_eclasses(root_package)
    node_type_classes = determine_node_type_classes(root_obj)
    association_info = classify_association_classes(eclasses, node_type_classes)
    association_eclass_set = set(association_info.keys())

    # --- eOpposite analysis --------------------------------------------------
    # For each reference, record whether it is bidirectional (has an eOpposite)
    # and, for opposite pairs, which side is PRIMARY (emitted + mirrored) vs
    # SUPPRESSED (skipped, since the primary already covers the logical link).
    # Keyed by (owner_class_name, ref_name).
    bidirectional_refs = set()   # refs that should get a synthesized reverse
    suppressed_ref_names = set() # partner side of a pair; skip entirely

    for c in eclasses:
        for ref in c.eAllReferences():
            opp = getattr(ref, "eOpposite", None)
            if opp is None:
                continue
            this_key = (c.name, ref.name)
            other_key = (ref.eType.name, opp.name)
            # Choose the primary side deterministically:
            #  - the containment side owns the object, so it is primary; else
            #  - the association-class side (ref lives on an association class)
            #    is primary; else
            #  - lexicographically smaller (class, ref) name is primary.
            this_is_assoc = c in association_eclass_set
            other_is_assoc = ref.eType in association_eclass_set
            if ref.containment and not opp.containment:
                primary, suppressed = this_key, other_key
            elif opp.containment and not ref.containment:
                primary, suppressed = other_key, this_key
            elif this_is_assoc and not other_is_assoc:
                primary, suppressed = this_key, other_key
            elif other_is_assoc and not this_is_assoc:
                primary, suppressed = other_key, this_key
            else:
                primary, suppressed = sorted([this_key, other_key])
            bidirectional_refs.add(primary)
            suppressed_ref_names.add(suppressed)

    all_objects = list(root_obj.eAllContents())

    # Only instances whose EClass is a node type become nodes.
    node_objects = [o for o in all_objects if o.eClass in node_type_classes]
    # Only instances whose EClass was classified as an association class.
    assoc_objects = [o for o in all_objects if o.eClass in association_eclass_set]
    # Anything else (nested classes that are neither) is intentionally
    # skipped — not a node, not an edge. Flag it so it isn't silently lost.
    handled_classes = node_type_classes | association_eclass_set
    unhandled_objects = [o for o in all_objects if o.eClass not in handled_classes]
    if unhandled_objects:
        unhandled_types = sorted({o.eClass.name for o in unhandled_objects})
        print(f"Warning: {len(unhandled_objects)} object(s) of type(s) "
              f"{unhandled_types} are neither node types nor recognized "
              f"association classes and were skipped.")

    # --- Build node index & per-type object lists --------------------------
    nodes_by_type = {}
    for obj in node_objects:
        nodes_by_type.setdefault(obj.eClass.name, []).append(obj)

    obj_index = {}  # id(obj) -> (type_name, index_within_type)
    for type_name, objs in nodes_by_type.items():
        for i, obj in enumerate(objs):
            obj_index[id(obj)] = (type_name, i)

    # --- Node features -------------------------------------------------------
    for type_name, objs in nodes_by_type.items():
        _, x = build_feature_matrix(objs)
        if x.shape[1] == 0:
            x = torch.zeros((len(objs), 1), dtype=torch.float)
        data[type_name].x = x
        data[type_name].num_nodes = len(objs)

    # --- Plain edges: any reference between two node-type objects that is --
    # --- NOT mediated by an association class (containment or not) ----------
    plain_edges = {}  # (src_type, ref_name, dst_type) -> ([src...], [dst...])
    plain_edge_bidir = {}  # same key -> bool (has eOpposite => bidirectional)

    for obj in node_objects:
        src_type, src_idx = obj_index[id(obj)]

        for ref in obj.eClass.eAllReferences():
            # Skip references pointing at association-class instances:
            # those are represented separately as featured edges below.
            if ref.eType in association_eclass_set:
                continue

            # Skip the suppressed side of an eOpposite pair, so a bidirectional
            # link is emitted once (from its primary side) and mirrored, rather
            # than appearing as two separate edge types.
            if (obj.eClass.name, ref.name) in suppressed_ref_names:
                continue

            try:
                value = getattr(obj, ref.name)
            except Exception:
                continue
            if value is None:
                continue

            is_bidir = (obj.eClass.name, ref.name) in bidirectional_refs

            targets = list(value) if ref.many else [value]
            for target in targets:
                if target is None or id(target) not in obj_index:
                    continue  # dangling ref, ignored root, or non-node type

                dst_type, dst_idx = obj_index[id(target)]
                key = (src_type, ref.name, dst_type)
                plain_edges.setdefault(key, ([], []))
                plain_edges[key][0].append(src_idx)
                plain_edges[key][1].append(dst_idx)
                plain_edge_bidir[key] = is_bidir

    for (src_type, rel_name, dst_type), (src_list, dst_list) in plain_edges.items():
        data[src_type, rel_name, dst_type].edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )
        # No edge features for these — structural edges only.

        # Reverse ONLY if the reference is bidirectional in the metamodel
        # (declares an eOpposite). Unidirectional refs stay directed.
        if plain_edge_bidir.get((src_type, rel_name, dst_type), False):
            data[dst_type, f"rev_{rel_name}", src_type].edge_index = torch.tensor(
                [dst_list, src_list], dtype=torch.long
            )

    # --- Association-class edges: owner -> target, features from AC's ------
    # --- own attributes -------------------------------------------------------
    assoc_by_class = {}
    for obj in assoc_objects:
        assoc_by_class.setdefault(obj.eClass, []).append(obj)

    for ac_class, ac_objs in assoc_by_class.items():
        info = association_info[ac_class]

        edge_src, edge_dst, kept_objs = [], [], []

        for ac_obj in ac_objs:
            owner_obj = ac_obj.eContainer()
            if owner_obj is None or id(owner_obj) not in obj_index:
                continue

            target_ref = info["target_ref"]
            try:
                target_value = getattr(ac_obj, target_ref.name)
            except Exception:
                continue
            if target_value is None:
                continue

            targets = list(target_value) if target_ref.many else [target_value]
            src_type, src_idx = obj_index[id(owner_obj)]

            for target_obj in targets:
                if target_obj is None or id(target_obj) not in obj_index:
                    continue
                dst_type, dst_idx = obj_index[id(target_obj)]

                edge_src.append(src_idx)
                edge_dst.append(dst_idx)
                kept_objs.append(ac_obj)

        if not edge_src:
            continue

        owner_type_name = info["owner_class"].name
        target_type_name = info["target_class"].name
        edge_type = (owner_type_name, ac_class.name, target_type_name)

        data[edge_type].edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

        # Association class's own attributes -> edge_attr (e.g. "hours")
        _, edge_attr = build_feature_matrix(kept_objs)
        if edge_attr.shape[1] > 0:
            data[edge_type].edge_attr = edge_attr

        # Bidirectional ONLY if the association class's TARGET reference declares
        # an eOpposite (bidirectional in the metamodel). e.g. Connection.room
        # has eOpposite room_connection -> reverse; Plays.game has none -> stay
        # directed (no rev_Plays).
        target_ref = info["target_ref"]
        target_is_bidir = getattr(target_ref, "eOpposite", None) is not None
        if target_is_bidir:
            rev_type = (target_type_name, f"rev_{ac_class.name}", owner_type_name)
            data[rev_type].edge_index = torch.tensor(
                [edge_dst, edge_src], dtype=torch.long
            )
            if edge_attr.shape[1] > 0:
                data[rev_type].edge_attr = edge_attr

    return data


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert all snapshot_*.xmi files in a folder into "
                    "PyTorch Geometric HeteroData graphs (snapshot_NNNN.xmi -> "
                    "hd_NNNN.pt)."
    )
    parser.add_argument(
        "--ecore", default="iotbuilding.ecore",
        help="Path to the .ecore metamodel file (default: iotbuilding.ecore)"
    )
    parser.add_argument(
        "--in-dir", default="xmi_new",
        help="Folder containing snapshot_*.xmi files (default: xmi_new)"
    )
    parser.add_argument(
        "--out-dir", default="pt_graphs",
        help="Folder to write hd_*.pt files into (default: pt_graphs)"
    )
    args = parser.parse_args()

    # Metamodel is loaded once and reused for every instance.
    rset, root_package = load_metamodel(args.ecore)

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "snapshot_*.xmi")))
    if not files:
        print(f"No snapshot_*.xmi files found in {args.in_dir}/")
        return

    for path in files:
        base = os.path.basename(path)               # snapshot_0000.xmi
        stem = os.path.splitext(base)[0]            # snapshot_0000
        suffix = stem[len("snapshot_"):]           # 0000
        out_path = os.path.join(args.out_dir, f"hd_{suffix}.pt")

        root_obj = load_model(rset, path)
        graph = build_hetero_graph(root_obj, root_package)
        torch.save(graph, out_path)
        print(f"{base} -> {out_path}")

    print(f"Converted {len(files)} file(s) into {args.out_dir}/")


if __name__ == "__main__":
    main()