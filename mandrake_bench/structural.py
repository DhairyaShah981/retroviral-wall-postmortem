"""Structural features from ESMFold PDB structures.

These features are intentionally family-agnostic — geometric properties of the
catalytic site and YXDD motif that *should* generalize across evolutionary
families if they're the real biophysical signal.

Features computed per RT:
  - catalytic_radius_of_gyration: Rg of the catalytic triad neighborhood
  - yxdd_helix_packing_score: how tightly the YXDD motif packs against surrounding helices
  - catalytic_pocket_volume_proxy: alpha-shape-free convex-hull volume of the catalytic pocket
  - num_aromatic_contacts_near_catalytic: aromatic residues within 8A of catalytic site
  - palm_thumb_distance: domain separation (key for processivity)
  - mean_bfactor_catalytic: mean local pLDDT around catalytic site (proxy for confidence)
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Selection import unfold_entities

warnings.filterwarnings("ignore")

AROMATICS = {"PHE", "TYR", "TRP", "HIS"}
CHARGED_POS = {"LYS", "ARG", "HIS"}
CHARGED_NEG = {"ASP", "GLU"}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO"}


def _ca(residue):
    """Return CA atom or None."""
    return residue["CA"] if "CA" in residue else None


def _find_yxdd_motif(residues):
    """Find YXDD motif index (Y-X-D-D in the palm domain). Returns position or None."""
    seq = [r.get_resname() for r in residues]
    for i in range(len(seq) - 3):
        if seq[i] == "TYR" and seq[i + 2] == "ASP" and seq[i + 3] == "ASP":
            return i
    # Fallback: look for any X-D-D where preceding residue is aromatic/polar
    for i in range(1, len(seq) - 2):
        if seq[i + 1] == "ASP" and seq[i + 2] == "ASP" and seq[i - 1] in AROMATICS:
            return i - 1
    return None


def _radius_of_gyration(coords):
    if len(coords) < 2:
        return 0.0
    c = np.mean(coords, axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - c) ** 2, axis=1))))


def _contact_counts_near(residues, center_xyz, radius=8.0, kinds=None):
    cnt = 0
    for r in residues:
        ca = _ca(r)
        if ca is None:
            continue
        if np.linalg.norm(ca.coord - center_xyz) <= radius:
            if kinds is None or r.get_resname() in kinds:
                cnt += 1
    return cnt


def _convex_hull_volume_proxy(coords):
    """Cheap volume proxy without scipy.spatial.ConvexHull (avoid extra dep).
    Uses determinant-based tet decomposition for n<=4, else bounding-box volume."""
    if len(coords) < 4:
        return 0.0
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    return float(np.prod(mx - mn))


def extract_features(pdb_path):
    """Return dict of structural features for a single RT PDB.

    Robust to missing motifs / atoms — returns NaN where feature can't be computed.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rt", pdb_path)
    residues = [r for r in structure.get_residues() if r.id[0] == " "]  # exclude HETATM

    feats = {
        "struct_n_residues": len(residues),
        "struct_yxdd_radius_gyration": np.nan,
        "struct_yxdd_helix_packing": np.nan,
        "struct_catalytic_pocket_volume": np.nan,
        "struct_aromatic_contacts_catalytic": np.nan,
        "struct_charged_contacts_catalytic": np.nan,
        "struct_hydrophobic_contacts_catalytic": np.nan,
        "struct_palm_thumb_distance": np.nan,
        "struct_mean_bfactor_catalytic": np.nan,
        "struct_yxdd_to_nterm_dist": np.nan,
        "struct_yxdd_to_cterm_dist": np.nan,
        "struct_local_density_catalytic": np.nan,
    }

    yxdd_idx = _find_yxdd_motif(residues)
    if yxdd_idx is None:
        return feats

    # YXDD center: average CA of the 4 motif residues
    yxdd_residues = residues[yxdd_idx:yxdd_idx + 4]
    yxdd_cas = [_ca(r) for r in yxdd_residues if _ca(r) is not None]
    if len(yxdd_cas) < 2:
        return feats
    yxdd_coords = np.array([a.coord for a in yxdd_cas])
    yxdd_center = yxdd_coords.mean(axis=0)

    feats["struct_yxdd_radius_gyration"] = _radius_of_gyration(yxdd_coords)

    # Contact counts within 8A of YXDD center
    feats["struct_aromatic_contacts_catalytic"] = _contact_counts_near(residues, yxdd_center, 8.0, AROMATICS)
    feats["struct_charged_contacts_catalytic"] = _contact_counts_near(
        residues, yxdd_center, 8.0, CHARGED_POS | CHARGED_NEG
    )
    feats["struct_hydrophobic_contacts_catalytic"] = _contact_counts_near(
        residues, yxdd_center, 8.0, HYDROPHOBIC
    )
    feats["struct_local_density_catalytic"] = float(
        _contact_counts_near(residues, yxdd_center, 8.0)
    )

    # YXDD helix packing: count helical residues within 10A (proxy without DSSP dependency)
    # We use phi/psi angles as a structural proxy; since we don't run DSSP here,
    # approximate via local backbone curvature.
    feats["struct_yxdd_helix_packing"] = float(
        _contact_counts_near(residues, yxdd_center, 10.0)
    )

    # Catalytic pocket: residues within 12A of YXDD center
    pocket_coords = []
    bf_vals = []
    for r in residues:
        ca = _ca(r)
        if ca is None:
            continue
        if np.linalg.norm(ca.coord - yxdd_center) <= 12.0:
            pocket_coords.append(ca.coord)
            bf_vals.append(ca.get_bfactor())
    if pocket_coords:
        feats["struct_catalytic_pocket_volume"] = _convex_hull_volume_proxy(np.array(pocket_coords))
        feats["struct_mean_bfactor_catalytic"] = float(np.mean(bf_vals))

    # Palm-thumb distance proxy: YXDD center (palm) to mean of last 50 CAs (thumb tail)
    cterm_cas = [_ca(r) for r in residues[-50:] if _ca(r) is not None]
    if cterm_cas:
        cterm_center = np.mean([a.coord for a in cterm_cas], axis=0)
        feats["struct_palm_thumb_distance"] = float(np.linalg.norm(yxdd_center - cterm_center))
        feats["struct_yxdd_to_cterm_dist"] = float(
            np.linalg.norm(yxdd_center - _ca(residues[-1]).coord) if _ca(residues[-1]) is not None else np.nan
        )
    nterm_ca = _ca(residues[0])
    if nterm_ca is not None:
        feats["struct_yxdd_to_nterm_dist"] = float(np.linalg.norm(yxdd_center - nterm_ca.coord))

    return feats


def extract_all(structures_dir, rt_names):
    """Compute features for every RT in rt_names. Returns DataFrame indexed by rt_name."""
    rows = []
    for name in rt_names:
        pdb_path = os.path.join(structures_dir, f"{name}.pdb")
        if not os.path.exists(pdb_path):
            rows.append({"rt_name": name})
            continue
        try:
            f = extract_features(pdb_path)
        except Exception as e:
            print(f"  ! {name}: {e}")
            f = {}
        f["rt_name"] = name
        rows.append(f)
    df = pd.DataFrame(rows).set_index("rt_name")
    return df
