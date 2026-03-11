#!/usr/bin/env python3
"""
DWI preprocessing pipeline.
Patched to handle odd dimensions for FSL topup/eddy compatibility.
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger("dwipreproc")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)


def run_cmd(cmd, cwd=None, timeout=None):
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info("CMD: %s", cmd_str)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    
    if result.returncode != 0:
        logger.error("Command failed (exit %d): %s", result.returncode, cmd_str)
        logger.error("STDOUT: %s", result.stdout) 
        logger.error("STDERR: %s", result.stderr)
        result.check_returncode()
    return result


def check_binaries(names):
    missing = [n for n in names if shutil.which(n) is None]
    if missing:
        raise RuntimeError(f"Required binaries not found on PATH: {missing}")


def find_eddy():
    for name in ["eddy_cuda", "eddy_openmp", "eddy"]:
        if shutil.which(name):
            logger.info("Using eddy binary: %s", name)
            return name
    raise RuntimeError("No eddy binary found (tried eddy_cuda, eddy_openmp, eddy)")


def find_bedpostx():
    for name in ["bedpostx_gpu", "bedpostx"]:
        if shutil.which(name):
            logger.info("Using bedpostx binary: %s", name)
            return name
    raise RuntimeError("No bedpostx binary found (tried bedpostx_gpu, bedpostx)")


def detect_bedpost_output_type():
    """Determine FSLOUTPUTTYPE for bedpostx outputs.

    Older probtrackx2_gpu versions are significantly faster reading
    uncompressed .nii files. Once FSL is upgraded to handle .nii.gz
    efficiently in probtrackx, this can return "NIFTI_GZ".
    """
    try:
        result = subprocess.run(
            ["probtrackx2_gpu", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        help_text = result.stdout + result.stderr
        if "--nifti_gz" in help_text or "--compressed" in help_text:
            logger.info("Using FSLOUTPUTTYPE=NIFTI_GZ (probtrackx2_gpu supports compressed)")
            return "NIFTI_GZ"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    logger.info("Using FSLOUTPUTTYPE=NIFTI for bedpostx (uncompressed for probtrackx performance)")
    return "NIFTI"


def step_done(output_path, force):
    if not force and Path(output_path).exists():
        logger.info("Skipping (already done): %s", output_path)
        return True
    return False


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def timed_step(name):
    logger.info("=== %s ===", name)
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    logger.info("=== %s completed in %.1fs ===", name, elapsed)


def load_bvals(path):
    return np.loadtxt(path).ravel()


def load_bvecs(path):
    bvecs = np.loadtxt(path)
    if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
        bvecs = bvecs.T
    return bvecs


def save_bvals(bvals, path):
    np.savetxt(path, bvals.reshape(1, -1), fmt="%g", delimiter=" ")


def save_bvecs(bvecs, path):
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T
    np.savetxt(path, bvecs, fmt="%.6f", delimiter=" ")


def pe_direction_to_vector(pe_dir):
    mapping = {
        "i":  "1 0 0",
        "i-": "-1 0 0",
        "j":  "0 1 0",
        "j-": "0 -1 0",
        "k":  "0 0 1",
        "k-": "0 0 -1",
    }
    if pe_dir not in mapping:
        raise ValueError(f"Unknown PhaseEncodingDirection: {pe_dir}")
    return mapping[pe_dir]


def is_negative_pe(pe_dir):
    return pe_dir.endswith("-")


def pe_axis(pe_dir):
    return pe_dir.rstrip("-")


def get_performance_cores():
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return int(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return os.cpu_count() or 1


def ensure_gz(src_path, dst_path):
    if src_path.endswith(".nii.gz"):
        shutil.copy2(src_path, dst_path)
    else:
        img = nib.load(src_path)
        nib.save(img, dst_path)


# ---------------------------------------------------------------------------
# Step 1: Input Validation
# ---------------------------------------------------------------------------

def validate_and_stage(input_path, output_dir):
    input_dir = Path(input_path)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input path is not a directory: {input_path}")

    # Only consider NIfTI files that have companion .json/.bval/.bvec files
    # (this filters out dtifit outputs like dwi_FA.nii.gz from prior runs)
    niis = []
    for f in sorted(input_dir.iterdir()):
        if not (f.name.endswith(".nii") or f.name.endswith(".nii.gz")):
            continue
        stem = f.name.replace(".nii.gz", "").replace(".nii", "")
        if all((input_dir / (stem + ext)).exists() for ext in [".json", ".bval", ".bvec"]):
            niis.append(f)
    if len(niis) != 2:
        raise ValueError(
            f"Expected exactly 2 NIfTI files (with .json/.bval/.bvec sidecars) "
            f"in {input_path}, found {len(niis)}: "
            + ", ".join(f.name for f in niis)
        )

    datasets = []
    for nii in niis:
        stem = nii.name.replace(".nii.gz", "").replace(".nii", "")
        companions = {}
        for ext in [".json", ".bval", ".bvec"]:
            companion = input_dir / (stem + ext)
            if not companion.exists():
                raise FileNotFoundError(
                    f"Missing companion file: {companion}"
                )
            companions[ext] = companion
        with open(companions[".json"]) as f:
            sidecar = json.load(f)
        datasets.append({
            "nii": nii,
            "stem": stem,
            "json_path": companions[".json"],
            "bval_path": companions[".bval"],
            "bvec_path": companions[".bvec"],
            "sidecar": sidecar,
        })

    sc_a, sc_b = datasets[0]["sidecar"], datasets[1]["sidecar"]
    readout_time = sc_a.get("TotalReadoutTime")
    pe_a = sc_a.get("PhaseEncodingDirection")
    pe_b = sc_b.get("PhaseEncodingDirection")

    if readout_time is None:
        raise ValueError(
            f"TotalReadoutTime missing from {datasets[0]['json_path']}. "
            "This field is required for topup/eddy."
        )
    if pe_a is None or pe_b is None:
        missing = [ds["json_path"] for ds, pe in [(datasets[0], pe_a), (datasets[1], pe_b)] if pe is None]
        raise ValueError(
            f"PhaseEncodingDirection missing from: {missing}"
        )

    img_a = nib.load(str(datasets[0]["nii"]))
    img_b = nib.load(str(datasets[1]["nii"]))
    shape_a = img_a.shape
    shape_b = img_b.shape

    if is_negative_pe(pe_a):
        dtiR_ds, dti_ds = datasets[0], datasets[1]
    else:
        dti_ds, dtiR_ds = datasets[0], datasets[1]

    pe_pos = dti_ds["sidecar"]["PhaseEncodingDirection"]
    pe_neg = dtiR_ds["sidecar"]["PhaseEncodingDirection"]

    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    file_map = {"dti": dti_ds, "dtiR": dtiR_ds}
    for prefix, ds in file_map.items():
        dst_nii = tmp_dir / f"{prefix}.nii.gz"
        ensure_gz(str(ds["nii"]), str(dst_nii))
        shutil.copy2(str(ds["json_path"]), str(tmp_dir / f"{prefix}.json"))
        shutil.copy2(str(ds["bval_path"]), str(tmp_dir / f"{prefix}.bval"))
        shutil.copy2(str(ds["bvec_path"]), str(tmp_dir / f"{prefix}.bvec"))

    return {
        "input_dir": input_dir,
        "tmp_dir": tmp_dir,
        "readout_time": readout_time,
        "pe_positive": pe_pos,
        "pe_negative": pe_neg,
        "spatial_dims": shape_a[:3],
        "n_vols_dti": shape_a[3] if dti_ds is datasets[0] else shape_b[3],
        "n_vols_dtiR": shape_b[3] if dtiR_ds is datasets[1] else shape_a[3],
    }

# ---------------------------------------------------------------------------
# Step 2: Topup + Eddy
# ---------------------------------------------------------------------------

def _set_eddy_outputs(info, eddy_out, tmp, brain_mask):
    """Populate info dict with eddy output paths."""
    info["eddy_nii"] = eddy_out + ".nii.gz"
    info["dwi_corrected"] = info["eddy_nii"]
    rotated = eddy_out + ".eddy_rotated_bvecs"
    info["rotated_bvecs"] = rotated if Path(rotated).exists() else str(tmp / "merged.bvec")
    info["merged_bval"] = str(tmp / "merged.bval")
    info["eddy_bvals"] = info["merged_bval"]
    info["brain_mask"] = brain_mask


def step_topup_eddy(info, b0tolerance, nthr, isPadOdd, force=False):
    tmp = info["tmp_dir"]
    eddy_out = str(tmp / "eddy")
    brain_mask = str(tmp / "brain_mask.nii.gz")

    # If eddy is already complete, just set output paths and skip everything
    if step_done(eddy_out + ".nii.gz", force):
        _set_eddy_outputs(info, eddy_out, tmp, brain_mask)
        return

    # --- Load both acquisitions once ---
    dti_bvals = load_bvals(str(tmp / "dti.bval"))
    dtiR_bvals = load_bvals(str(tmp / "dtiR.bval"))

    dti_img = nib.load(str(tmp / "dti.nii.gz"))
    dti_data = dti_img.get_fdata(dtype=np.float32)
    dtiR_img = nib.load(str(tmp / "dtiR.nii.gz"))
    dtiR_data = dtiR_img.get_fdata(dtype=np.float32)

    # --- Padding logic for odd dimensions ---
    dims = list(info["spatial_dims"])
    if isPadOdd:
        needs_padding = any(d % 2 != 0 for d in dims)
        if needs_padding:
            new_dims = [d + 1 if d % 2 != 0 else d for d in dims]
            logger.info(f"Odd dimensions detected {dims}. Padding to {new_dims}...")
            pad_width = [(0, 1 if d % 2 != 0 else 0) for d in dims] + [(0, 0)]
            dti_data = np.pad(dti_data, pad_width, mode='constant', constant_values=0)
            dtiR_data = np.pad(dtiR_data, pad_width, mode='constant', constant_values=0)
            nib.save(nib.Nifti1Image(dti_data, dti_img.affine, dti_img.header),
                     str(tmp / "dti.nii.gz"))
            nib.save(nib.Nifti1Image(dtiR_data, dtiR_img.affine, dtiR_img.header),
                     str(tmp / "dtiR.nii.gz"))
            info["spatial_dims"] = tuple(new_dims)
            dims = new_dims

    # --- Extract b0 volumes (single load, no subprocess calls) ---
    dti_b0_indices = np.where(dti_bvals < b0tolerance)[0]
    dtiR_b0_indices = np.where(dtiR_bvals < b0tolerance)[0]

    b0_vols = np.concatenate([
        dti_data[..., dti_b0_indices],
        dtiR_data[..., dtiR_b0_indices],
    ], axis=3)

    b0_all = str(tmp / "b0_all.nii.gz")
    nib.save(nib.Nifti1Image(b0_vols, dti_img.affine, dti_img.header), b0_all)
    del dti_data, dtiR_data, b0_vols

    # --- Acq params ---
    vec_pos = pe_direction_to_vector(info["pe_positive"])
    vec_neg = pe_direction_to_vector(info["pe_negative"])
    readout = info["readout_time"]
    acq_param = str(tmp / "acq_param.txt")
    with open(acq_param, "w") as f:
        f.write(f"{vec_pos} {readout}\n{vec_neg} {readout}\n")

    acq_param_topup = str(tmp / "acq_param_topup.txt")
    with open(acq_param_topup, "w") as f:
        for _ in dti_b0_indices: f.write(f"{vec_pos} {readout}\n")
        for _ in dtiR_b0_indices: f.write(f"{vec_neg} {readout}\n")

    # --- Config choice ---
    if all(d % 4 == 0 for d in dims): cnf = "b02b0_4.cnf"
    elif all(d % 2 == 0 for d in dims): cnf = "b02b0.cnf"
    else: cnf = "b02b0_1.cnf"

    # --- Run topup ---
    topup_out = str(tmp / "topup_results")
    if not step_done(topup_out + "_fieldcoef.nii.gz", force):
        with timed_step("Topup"):
            run_cmd([
                "topup",
                f"--imain={b0_all}",
                f"--datain={acq_param_topup}",
                f"--config={cnf}",
                f"--out={topup_out}",
                f"--iout={str(tmp / 'topup_b0')}",
                f"--nthr={nthr}"
            ])
    # --- Brain extraction & Eddy ---
    topup_b0_mean = str(tmp / "topup_b0_mean.nii.gz")
    if not step_done(brain_mask, force):
        run_cmd(["fslmaths", str(tmp / "topup_b0.nii.gz"), "-Tmean", topup_b0_mean])
        run_cmd(["bet", topup_b0_mean, str(tmp / "brain"), "-f", "0.25", "-R", "-m"])

    merged_nii = str(tmp / "merged.nii.gz")
    run_cmd(["fslmerge", "-t", merged_nii, str(tmp / "dti.nii.gz"), str(tmp / "dtiR.nii.gz")])
    index_file = str(tmp / "index.txt")
    with open(index_file, "w") as f:
        f.write(" ".join(["1"] * info["n_vols_dti"] + ["2"] * info["n_vols_dtiR"]) + "\n")

    all_bvals = np.concatenate([dti_bvals, dtiR_bvals])
    rounded_bvals = (np.round(all_bvals / 100.0) * 100).astype(int)
    save_bvals(rounded_bvals, str(tmp / 'merged.bval'))
    save_bvecs(np.hstack([load_bvecs(str(tmp / "dti.bvec")), load_bvecs(str(tmp / "dtiR.bvec"))]), str(tmp / "merged.bvec"))
    with timed_step("Eddy"):
        run_cmd([
            find_eddy(),
            f"--imain={merged_nii}",
            f"--mask={brain_mask}",
            f"--acqp={acq_param}",
            f"--index={index_file}",
            f"--bvecs={str(tmp / 'merged.bvec')}",
            f"--bvals={str(tmp / 'merged.bval')}",
            f"--topup={topup_out}",
            "--data_is_shelled",
            "--repol",
            f"--out={eddy_out}"
        ])

    _set_eddy_outputs(info, eddy_out, tmp, brain_mask)

# ---------------------------------------------------------------------------
# Step 3: DTI Fit
# ---------------------------------------------------------------------------

def step_dtifit(info, base_name, force=False):
    logger.info("=== DTI Fit ===")
    out_prefix = base_name
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    fa_path = out_prefix + "_FA.nii.gz"
    if step_done(fa_path, force):
        info["fa_path"] = fa_path
        info["fa_thr_path"] = out_prefix + "_FA_thr.nii.gz"
        if not Path(info["fa_thr_path"]).exists():
            run_cmd(["fslmaths", fa_path, "-ero", "-thr", "0.15", "-bin", info["fa_thr_path"]])
        return
    run_cmd(["dtifit", f"--data={info['eddy_nii']}", f"--out={out_prefix}", f"--mask={info['brain_mask']}", f"--bvecs={info['rotated_bvecs']}", f"--bvals={info['merged_bval']}"])
    fa_thr = out_prefix + "_FA_thr.nii.gz"
    run_cmd(["fslmaths", fa_path, "-ero", "-thr", "0.15", "-bin", fa_thr])
    info["fa_path"] = fa_path
    info["fa_thr_path"] = fa_thr

# ---------------------------------------------------------------------------
# Step 4: Rim Clean
# ---------------------------------------------------------------------------

def step_rim_clean(info, output_dir, base_name, force=False):
    logger.info("=== Rim Clean ===")
    fa_path = info["fa_path"]
    cleaned_fa = str(Path(output_dir) / f"r{Path(base_name).name}_FA.nii.gz")

    if step_done(cleaned_fa, force):
        info["cleaned_fa_path"] = cleaned_fa
        return

    img = nib.load(fa_path)
    data = img.get_fdata(dtype=np.float32)
    cleaned = data.copy()

    mask = (data > 0).astype(np.uint8)
    dist_map = distance_transform_edt(mask)

    total_flagged = 0
    for iteration in range(1):
        outer_dist = float(iteration + 1)
        inner_dist = float(iteration + 2)

        outer_band = dist_map == outer_dist
        inner_band = dist_map == inner_dist
        n_outer = int(outer_band.sum())
        logger.info("Iteration %d/1 (shell dist=%d vs dist=%d) — %d rim voxels",
                    iteration + 1, int(outer_dist), int(inner_dist), n_outer)

        inner_indices = set(zip(*np.where(inner_band)))
        outer_coords = np.argwhere(outer_band)
        artefact_mask = np.zeros(data.shape, dtype=bool)

        for (x, y, z) in outer_coords:
            outer_val = cleaned[x, y, z]
            if outer_val == 0.0:
                continue
            inner_vals = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if (nx, ny, nz) in inner_indices:
                            inner_vals.append(cleaned[nx, ny, nz])
            if not inner_vals:
                artefact_mask[x, y, z] = True
                continue
            if outer_val > np.median(inner_vals):
                artefact_mask[x, y, z] = True

        n_flagged = int(artefact_mask.sum())
        logger.info("Artefacts flagged: %d / %d (%.1f%%)", n_flagged, n_outer,
                    100.0 * n_flagged / n_outer if n_outer > 0 else 0.0)
        cleaned[artefact_mask] = 0.0
        total_flagged += n_flagged

    logger.info("Total voxels zeroed: %d", total_flagged)
    out_img = nib.Nifti1Image(cleaned, img.affine, img.header)
    nib.save(out_img, cleaned_fa)
    info["cleaned_fa_path"] = cleaned_fa

# ---------------------------------------------------------------------------
# Step 5: MMORF Atlas Registration
# ---------------------------------------------------------------------------

def step_mmorf_atlas(info, output_dir, nthr, template_path=None, atlas_path=None, force=False):
    tmp = Path(info["tmp_dir"])
    out_atlas = str(Path(output_dir) / "wHarvardOxford.nii.gz")
    warped_template = str(tmp / "wFMRIB58.nii.gz")

    if step_done(out_atlas, force):
        info["atlas_native"] = out_atlas
        return

    fsld = os.environ.get("FSLDIR")
    if template_path is None:
        if fsld:
            template_path = os.path.join(fsld, "data", "standard", "FMRIB58_FA_1mm.nii.gz")
        else:
            raise RuntimeError("FSLDIR not set and no template_path provided")
    if atlas_path is None:
        if fsld:
            atlas_path = os.path.join(fsld, "data", "atlases", "HarvardOxford",
                                      "HarvardOxford-cort-maxprob-thr25-1mm.nii.gz")
        else:
            raise RuntimeError("FSLDIR not set and no atlas_path provided")

    if not Path(template_path).exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    if not Path(atlas_path).exists():
        raise FileNotFoundError(f"Atlas not found: {atlas_path}")

    target = info["cleaned_fa_path"]
    affine_mat = str(tmp / "temp_affine.mat")
    identity_mat = str(tmp / "identity.mat")
    config_file = str(tmp / "mmorf_config.ini")
    warp_field_base = str(tmp / "warp_field")
    warp_field = warp_field_base + ".nii.gz"

    # Identity matrix
    if not Path(identity_mat).exists():
        with open(identity_mat, "w") as f:
            f.write("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1")

    # Linear registration (FLIRT)
    if not Path(affine_mat).exists():
        logger.info("=== FLIRT linear registration ===")
        run_cmd(["flirt", "-in", template_path, "-ref", target,
                 "-omat", affine_mat, "-dof", "12"])

    # MMORF config
    config_content = f"""\
warp_res_init           = 32
warp_scaling            = 1 1 2 2 2
img_warp_space          = {target}
lambda_reg              = 4.0e5 3.7e-1 3.1e-1 2.6e-1 2.2e-1
hires                   = 6
optimiser_max_it_lowres = 5
optimiser_max_it_hires  = 5
img_ref_scalar      = {target}
img_mov_scalar      = {template_path}
aff_ref_scalar      = {identity_mat}
aff_mov_scalar      = {affine_mat}
use_implicit_mask   = 0
use_mask_ref_scalar = 0 0 0 0 0
use_mask_mov_scalar = 0 0 0 0 0
mask_ref_scalar     = NULL
mask_mov_scalar     = NULL
fwhm_ref_scalar     = 8.0 8.0 4.0 2.0 1.0
fwhm_mov_scalar     = 8.0 8.0 4.0 2.0 1.0
lambda_scalar       = 1 1 1 1 1
estimate_bias       = 0
bias_res_init       = 32
lambda_bias_reg     = 1e9 1e9 1e9 1e9 1e9
warp_out            = {warp_field_base}
"""
    with open(config_file, "w") as f:
        f.write(config_content)

    # Non-linear registration (MMORF)
    with timed_step("MMORF non-linear registration"):
        run_cmd(["mmorf", "--config", config_file, "--num_threads", str(nthr)])

    # Apply warp to template (spline)
    run_cmd(["applywarp", f"--in={template_path}", f"--ref={target}",
             f"--warp={warp_field}", f"--premat={affine_mat}",
             f"--out={warped_template}", "--interp=spline"])

    # Apply warp to atlas (nearest neighbour)
    run_cmd(["applywarp", f"--in={atlas_path}", f"--ref={target}",
             f"--warp={warp_field}", f"--premat={affine_mat}",
             f"--out={out_atlas}", "--interp=nn"])

    info["atlas_native"] = out_atlas
    logger.info("Warped atlas written to: %s", out_atlas)


# ---------------------------------------------------------------------------
# Step 6: Bedpost
# ---------------------------------------------------------------------------

def run_bedpost_step(info, input_dir, force=False):
    with timed_step("Bedpost"):
        bed_dir = input_dir / "bedpost"
        bed_dir_x = input_dir / "bedpost.bedpostX"
        bed_done = bed_dir_x / "xfms" / "eye.mat"

        if step_done(str(bed_done), force):
            info["bedpostx_dir"] = str(bed_dir_x)
            return

        if force and bed_dir.exists():
            shutil.rmtree(bed_dir)
        if force and bed_dir_x.exists():
            shutil.rmtree(bed_dir_x)

        ensure_dir(bed_dir)
        shutil.copy2(info["dwi_corrected"], str(bed_dir / "data.nii.gz"))
        shutil.copy2(info["rotated_bvecs"], str(bed_dir / "bvecs"))
        shutil.copy2(info["eddy_bvals"], str(bed_dir / "bvals"))
        shutil.copy2(info["fa_thr_path"], str(bed_dir / "nodif_brain_mask.nii.gz"))

        bedpostx_exe = find_bedpostx()
        logger.info("Running bedpostx (%s) - this may take hours...", bedpostx_exe)
        old_output_type = os.environ.get("FSLOUTPUTTYPE")
        os.environ["FSLOUTPUTTYPE"] = detect_bedpost_output_type()
        try:
            run_cmd([bedpostx_exe, str(bed_dir)], timeout=172800)
        finally:
            if old_output_type is None:
                os.environ.pop("FSLOUTPUTTYPE", None)
            else:
                os.environ["FSLOUTPUTTYPE"] = old_output_type

        if not bed_done.exists():
            raise RuntimeError(
                f"bedpostx did not produce completion marker: {bed_done}"
            )

        info["bedpostx_dir"] = str(bed_dir_x)
        logger.info("Bedpost complete: %s", bed_dir_x)

# ---------------------------------------------------------------------------
# Step 7: Extract Masks
# ---------------------------------------------------------------------------

def step_extract_masks(info, output_dir, force=False):
    logger.info("=== Extract Masks ===")
    masks_dir = ensure_dir(Path(output_dir) / "masks")
    atlas_path = info["atlas_native"]

    img = nib.load(atlas_path)
    data = np.round(img.get_fdata()).astype(int)

    indices = sorted(set(data[data > 0].ravel()))

    created = 0
    skipped = 0
    for idx in indices:
        mask_path = str(masks_dir / f"{idx}.nii")
        if step_done(mask_path, force):
            skipped += 1
            continue
        mask_data = (data == idx).astype(np.uint8)
        if mask_data.sum() == 0:
            continue
        mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
        mask_img.set_data_dtype(np.uint8)
        nib.save(mask_img, mask_path)
        created += 1

    logger.info("Created %d mask files (%d skipped) in %s", created, skipped, masks_dir)
    info["masks_dir"] = str(masks_dir)

# ---------------------------------------------------------------------------
# Step 8: Probtrackx
# ---------------------------------------------------------------------------

def has_seedlist_support(executable):
    """Check if the provided probtrackx2_gpu executable supports --seedlist."""
    try:
        # We now use the explicit path passed from the main step
        result = subprocess.run(
            [executable, "--help"],
            capture_output=True, text=True, timeout=10,
        )
        return "--seedlist" in result.stdout or "--seedlist" in result.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def step_probtrackx(info, output_dir, force=False):
    with timed_step("Probtrackx"):
        # 1. Resolve the correct binary at the very start
        fsld = os.environ.get("FSLDIR")
        if not fsld:
            raise EnvironmentError("FSLDIR environment variable not set.")
        
        # Explicitly target the binary in 'bin' to bypass buggy 'share' wrappers
        probtrackx_bin = os.path.join(fsld, "bin", "probtrackx2_gpu")

        masks_dir = Path(info["masks_dir"])
        bedpostx_dir = Path(info["bedpostx_dir"])
        probtrackx_dir = ensure_dir(Path(output_dir) / "probtrackx")
        log_dir = ensure_dir(probtrackx_dir / "logs") 

        merged = str(bedpostx_dir / "merged")
        nodif_mask = str(bedpostx_dir / "nodif_brain_mask")

        mask_files = sorted(masks_dir.glob("*.nii"), key=lambda p: int(p.stem))

        pending = []
        for mask_file in mask_files:
            idx = mask_file.stem
            out_dir = probtrackx_dir / idx
            # probtrackx2_gpu --seedlist appends '+' to the output dir name
            out_dir_plus = probtrackx_dir / f"{idx}+"
            fdt_paths = out_dir / "fdt_paths.nii.gz"
            fdt_paths_plus = out_dir_plus / "fdt_paths.nii.gz"
            if not (step_done(str(fdt_paths), force) or step_done(str(fdt_paths_plus), force)):
                ensure_dir(out_dir)
                pending.append((mask_file, out_dir))

        if not pending:
            logger.info("All probtrackx seeds already complete")
            return

        # 2. Pass the resolved binary to the support check
        if has_seedlist_support(probtrackx_bin):
            seedlist_file = probtrackx_dir / "seedlist.txt"
            with open(seedlist_file, "w") as f:
                for mask_file, out_dir in pending:
                    f.write(f"{mask_file} {out_dir}\n")
            
            logger.info("Running probtrackx2_gpu --seedlist (%d seeds)", len(pending))
            run_cmd([
                probtrackx_bin, # Use resolved binary
                f"--seedlist={seedlist_file}",
                "-P", "5000",
                "-s", merged,
                "-m", nodif_mask,
                "--opd", "--pd", "-l", "-c", "0.2", "--distthresh=0",
            ], cwd=log_dir)
        else:
            logger.info("--seedlist not supported, falling back to serial fsl_sub")
            commands = []
            for mask_file, out_dir in pending:
                # 3. Use the resolved binary in the task list
                cmd = (f"{probtrackx_bin} -x {mask_file} "
                       f"--dir={out_dir} --forcedir "
                       f"-P 5000 -s {merged} -m {nodif_mask} "
                       f"--opd --pd -l -c 0.2 --distthresh=0")
                commands.append(cmd)

            cmd_file = Path(output_dir) / "probtrackx_commands.txt"
            with open(cmd_file, "w") as f:
                f.write("\n".join(commands) + "\n")
            
            run_cmd(["fsl_sub", "-l", ".", "-N", "probtrackx",
                     "-T", "1", "-t", str(cmd_file)], cwd=log_dir)

# ---------------------------------------------------------------------------
# Step 9: Fiber Quantify
# ---------------------------------------------------------------------------

def step_fiber_quantify(info, output_dir, num_samples=5000, force=False):
    logger.info("=== Fiber Quantify ===")
    masks_dir = Path(info["masks_dir"])
    probtrackx_dir = Path(output_dir) / "probtrackx"
    quant_dir = ensure_dir(Path(output_dir) / "connectivity")

    density_npy = quant_dir / "density.npy"
    if step_done(str(density_npy), force):
        return

    # Discover regions and check that probtrackx has completed
    mask_files = sorted(masks_dir.glob("*.nii"), key=lambda p: int(p.stem))
    indices = [int(f.stem) for f in mask_files]
    n = len(indices)
    idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}

    # probtrackx2_gpu --seedlist appends '+' to output dir names;
    # build a mapping from region index to whichever path actually exists.
    fdt_map = {}
    missing = []
    for idx in indices:
        fdt = probtrackx_dir / str(idx) / "fdt_paths.nii.gz"
        fdt_plus = probtrackx_dir / f"{idx}+" / "fdt_paths.nii.gz"
        if fdt.exists():
            fdt_map[idx] = fdt
        elif fdt_plus.exists():
            fdt_map[idx] = fdt_plus
        else:
            missing.append(idx)
    if missing:
        logger.warning("Probtrackx incomplete (%d/%d regions missing) — "
                       "skipping fiber quantify. Re-run after probtrackx finishes.",
                       len(missing), n)
        return

    # Pre-load all masks and probtrackx results as flat arrays
    masks = {}
    mask_nvox = {}
    probs = {}

    for idx in indices:
        mask_data = nib.load(str(masks_dir / f"{idx}.nii")).get_fdata().ravel()
        masks[idx] = mask_data
        mask_nvox[idx] = int((mask_data > 0).sum())

        prob_data = nib.load(
            str(fdt_map[idx])
        ).get_fdata().ravel()
        probs[idx] = prob_data

    # Initialise symmetric NxN matrices (1 on diagonal, like Matlab)
    density_mat = np.eye(n)
    fiber_count_mat = np.eye(n)
    mean_mat = np.eye(n)
    max_mat = np.eye(n)

    logger.info("Computing pairwise connectivity for %d regions", n)

    for ii in range(n):
        i = indices[ii]
        pi = idx_to_pos[i]
        for jj in range(ii + 1, n):
            j = indices[jj]
            pj = idx_to_pos[j]

            # prob(i) values within mask(j)
            vals_ij = probs[i][masks[j] > 0]
            ij_mean = vals_ij.mean() if len(vals_ij) > 0 else 0.0
            ij_max = vals_ij.max() if len(vals_ij) > 0 else 0.0

            # prob(j) values within mask(i)
            vals_ji = probs[j][masks[i] > 0]
            ji_mean = vals_ji.mean() if len(vals_ji) > 0 else 0.0
            ji_max = vals_ji.max() if len(vals_ji) > 0 else 0.0

            # mean_mat / max_mat: mean/max of all values in masked region
            mean_mat[pi, pj] = ij_mean + ji_mean
            mean_mat[pj, pi] = mean_mat[pi, pj]
            max_mat[pi, pj] = ij_max + ji_max
            max_mat[pj, pi] = max_mat[pi, pj]

            # density / fiber_count: nonzero mean (fslstats -M equivalent)
            nz_ij = vals_ij[vals_ij != 0]
            nz_ji = vals_ji[vals_ji != 0]
            ij_nz_mean = nz_ij.mean() if len(nz_ij) > 0 else 0.0
            ji_nz_mean = nz_ji.mean() if len(nz_ji) > 0 else 0.0

            ij_sum = ij_nz_mean * mask_nvox[j]
            ji_sum = ji_nz_mean * mask_nvox[i]
            fc = ij_sum + ji_sum
            nf = (mask_nvox[i] + mask_nvox[j]) * (num_samples + 1)

            density_mat[pi, pj] = fc / nf if nf > 0 else 0.0
            density_mat[pj, pi] = density_mat[pi, pj]
            fiber_count_mat[pi, pj] = fc
            fiber_count_mat[pj, pi] = fc

    # Clean non-finite values
    for mat in [density_mat, fiber_count_mat, mean_mat, max_mat]:
        mat[~np.isfinite(mat)] = 0.0

    # Save as .npy and .tsv
    labels = [str(i) for i in indices]
    header = "\t".join([""] + labels)

    for name, mat in [("density", density_mat), ("fiber_count", fiber_count_mat),
                      ("mean", mean_mat), ("max", max_mat)]:
        np.save(str(quant_dir / f"{name}.npy"), mat)
        with open(quant_dir / f"{name}.tsv", "w") as f:
            f.write(header + "\n")
            for row in range(n):
                vals = "\t".join(f"{v:.6g}" for v in mat[row])
                f.write(f"{labels[row]}\t{vals}\n")

    logger.info("Connectivity matrices (%dx%d) written to %s", n, n, quant_dir)

# ---------------------------------------------------------------------------
# Generate README
# ---------------------------------------------------------------------------

def generate_readme(output_dir):
    log_path = Path(output_dir) / "dwipreproc.txt"
    readme_path = Path(output_dir) / "README.md"

    if not log_path.exists():
        return

    pattern = re.compile(r"=== (.+?) completed in ([\d.]+)s ===")
    rows = []

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                name = m.group(1)
                seconds = round(float(m.group(2)))
                rows.append((name, seconds))

    total_row = None
    other_rows = []
    for name, seconds in rows:
        if name.upper() == "TOTAL":
            total_row = (name, seconds)
        else:
            other_rows.append((name, seconds))

    rows = other_rows
    if total_row:
        rows.append(total_row)

    if not rows:
        return

    name_width = max([len(r[0]) for r in rows] + [len("Stage")])
    sec_width = max([len(str(r[1])) for r in rows] + [len("Seconds")])

    lines = [
        f"| {'Stage':<{name_width}} | {'Seconds':>{sec_width}} |",
        f"| {'-' * name_width} | {'-' * sec_width} |",
    ]
    for name, seconds in rows:
        lines.append(f"| {name:<{name_width}} | {seconds:>{sec_width}} |")

    with open(readme_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("README written to: %s", readme_path)

def main():
    parser = argparse.ArgumentParser(description="DWI preprocessing pipeline (topup, eddy, dtifit)")
    parser.add_argument("input_path", help="Directory containing exactly 2 DWI NIfTI files")
    parser.add_argument("output_path", nargs="?", default=None,
                        help="Output directory (default: <input_path>_YYYYMMDD)")
    parser.add_argument("--no-pad-odd", action="store_true", default=False,
                        help="Disable padding of odd dimensions for topup compatibility")
    parser.add_argument("--b0tolerance", type=float, default=40)
    parser.add_argument("--keep-temp", action="store_true", default=False,
                        help="Keep temporary files after pipeline completes")
    parser.add_argument("--baseName", type=str, default="dwi")
    parser.add_argument(
        "--no-bedpost",
        action="store_true",
        help="Disable bedpostx (runs by default)"
    )
    parser.add_argument("--force", action="store_true", default=False,
                        help="Re-run steps even if outputs already exist")
    parser.add_argument("-nthr", type=int, default=0)
    args = parser.parse_args()

    from datetime import date
    input_dir = Path(args.input_path).resolve()
    if args.output_path is None:
        output_dir = input_dir.parent / f"{input_dir.name}_{date.today().strftime('%Y%m%d')}"
    else:
        output_dir = Path(args.output_path).resolve()
    if input_dir == output_dir:
        raise ValueError("input_path and output_path must be different directories (no in-place writes).")

    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check write permission
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"No write permission for output directory: {output_dir}")
    
    setup_logging(output_dir / "dwipreproc.txt")

    if args.nthr < 1: args.nthr = max(1, multiprocessing.cpu_count() - 1)
    
    check_binaries(["probtrackx2_gpu", "eddy", "flirt", "bet", "topup", "fslmaths", "fslmerge",
                    "dtifit", "mmorf", "applywarp", "fsl_sub"])
    if not args.no_bedpost:
        find_bedpostx()

    with timed_step("TOTAL"):
        logger.info("=== Input Validation ===")
        info = validate_and_stage(str(input_dir), output_dir)

        step_topup_eddy(info, args.b0tolerance, args.nthr, not args.no_pad_odd, args.force)
        base_prefix = output_dir / args.baseName
        step_dtifit(info, str(base_prefix), args.force)

        step_rim_clean(info, output_dir, str(base_prefix), args.force)
        step_mmorf_atlas(info, output_dir, args.nthr, force=args.force)

        if not args.no_bedpost:
            run_bedpost_step(info, output_dir, args.force)

        step_extract_masks(info, output_dir, args.force)

        if not args.no_bedpost:
            step_probtrackx(info, output_dir, args.force)
            step_fiber_quantify(info, output_dir, force=args.force)

        if not args.keep_temp:
            td = info.get("tmp_dir")
            if td and Path(td).exists():
                shutil.rmtree(str(td))

    generate_readme(output_dir)

if __name__ == "__main__":
    main()