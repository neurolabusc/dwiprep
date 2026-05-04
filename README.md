# DWI Preprocessing Pipeline

A Python pipeline for processing diffusion-weighted MRI data using FSL tools. Takes a directory containing two DWI NIfTI files acquired with opposite phase-encoding directions (AP/PA) and runs a full preprocessing and analysis workflow.

## Installation

The pipeline requires a standard FSL installation along with a compatible graphics card and the corresponding accelerated tools, specifically mmorf, probtrackx2_gpu, bedpostx_gpu, and eddy_cuda. While NVIDIA users typically receive these via the standard FSL installer, macOS users must install the specific Metal-accelerated package to utilize eddy_metal on Apple Silicon. Python dependencies can be installed using pip install nibabel numpy scipy.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 dwipreproc.py <input_dir> [output_dir] [options]
```

If `output_dir` is omitted, defaults to `<input_dir>_YYYYMMDD`.

## Examples

A minimal example is the `quick` demo which checks that everything is installed and running. This is a toy dataset which is not realistic of real world data.

```bash
python dwipreproc.py ./quick
```

The `dwi` demo is more typical of real world data.

```bash
python dwipreproc.py ./dwi
```

By default, the pipeline allocates `N-1` logical processors (where `N` is the total system count) to balance high performance with system responsiveness.

For clustered computer nodes or shared environments, you may wish to explicitly limit this value to avoid resource contention. While additional cores follow Amdahl's Law—yielding diminishing returns as counts increase—providing a moderate number of threads is essential for maintaining high GPU utilization. You can explicitly set the thread count using the `-nthr` flag.

However, Eddy requires special consideration. The original FSL Eddy implementation for GPU does not support the `--nthr` option and is restricted to a single thread. In contrast, the optimized version allows for multi-threading, providing dramatic performance benefits that scale effectively up to the number of shells in your diffusion dataset. With the optimized version, if a thread count is not explicitly provided, the software will automatically determine an allocation based on available CPU threads and system RAM.

Due to these differences in Eddy versions, you should choose your command based on which implementation is available on your system.

For the original Eddy implementation, use:

```bash
python dwipreproc.py -nthr 4 ./dwi 
```

For the optimized Eddy implementation, use:

```bash
python dwipreproc.py -nthr 5 -nthrEddy 4 ./quick
```

# Performance

Several of these tests use optimized versions of mmorf, bedpost, probtrackx and eddy, for these we refer to the build date of these unreleased versions.

The `dwi` dataset provides a benchmark for processing a 140×140×81 diffusion dataset with 204 volumes. 

```
python dwipreproc.py --keep-temp ./dwi ./my_benchmark_results
```
Here is the performance for version 20260311 on an 64Gb Apple M5 Max:

| Stage                         | Seconds |
| ----------------------------- | ------- |
| Topup                         |     334 |
| Eddy                          |     340 |
| MMORF non-linear registration |      39 |
| Bedpost                       |      58 |
| Probtrackx                    |      67 |
| TOTAL                         |     894 |


Here is the performance for version 20260311 on an 48Gb Apple M4 Pro:

| Stage                         | Seconds |
| ----------------------------- | ------- |
| Topup                         |     327 |
| Eddy                          |     509 |
| MMORF non-linear registration |      79 |
| Bedpost                       |     109 |
| Probtrackx                    |     111 |
| TOTAL                         |    1192 |


Here is the performance for version 20260423 on a 128Gb Apple M3 Max (16 CPU / 40 GPU cores):

| Stage                         | Seconds |
| ----------------------------- | ------- |
| Topup                         |     387 |
| Eddy                          |     434 |
| MMORF non-linear registration |      54 |
| Bedpost                       |      68 |
| Probtrackx                    |      93 |
| TOTAL                         |    1115 |


Here is the performance for version 20260423 on a 16Gb Apple M1 (8 CPU / 8 GPU cores):

| Stage                         | Seconds |
| ----------------------------- | ------- |
| Topup                         |     675 |
| Eddy                          |    1640 |
| MMORF non-linear registration |     678 |
| Bedpost                       |     461 |
| Probtrackx                    |     390 |
| TOTAL                         |    3949 |


Here is the same test on a 128 Gb DGX Spark using the executables from FSL 6.0.7.21.

| Stage                         | Seconds |
| ----------------------------- | ------- |
| Topup                         |     331 |
| Eddy                          |    2901 |
| MMORF non-linear registration |     147 |
| Bedpost                       |     596 |
| Probtrackx                    |     300 |
| TOTAL                         |    4321 |

Here is the same test on the same 128 Gb DGX Spark using release 20260311.

| Stage                         | Seconds |
| ----------------------------- | ------- |
| Topup                         |     336 |
| Eddy                          |     905 |
| MMORF non-linear registration |     128 |
| Bedpost                       |     114 |
| Probtrackx                    |      97 |
| TOTAL                         |    1622 |

Here is the same test on a 128 Gb AMD 7995wx (96 cores, 192 threads) with RTX4090 GPU using the executables from FSL 6.0.7.21.

| Stage                         | Seconds |
| ----------------------------- | ------- |
| Topup                         |     441 |
| Eddy                          |    1102 |
| MMORF non-linear registration |     130 |
| Bedpost                       |     267 |
| Probtrackx                    |     162 |
| TOTAL                         |    2159 |

Here is the same test on the same 128 Gb 7995wx using release 20260311.

| Stage                         | Seconds |
| ----------------------------- | ------- |
| Topup                         |     448 |
| Eddy                          |     426 |
| MMORF non-linear registration |     116 |
| Bedpost                       |      64 |
| Probtrackx                    |      37 |
| TOTAL                         |    1135 |

### Options

| Flag | Description |
| --- | --- |
| `--force` | Re-run all steps even if outputs already exist |
| `--no-bedpost` | Skip bedpostx and downstream tractography steps |
| `--no-pad-odd` | Disable padding of odd spatial dimensions |
| `--keep-temp` | Keep temporary files after pipeline completes |
| `-nthr N` | Number of threads (default: CPU count - 1) |
| `--baseName NAME` | Base name for output files (default: `dwi`) |
| `--b0tolerance VAL` | b-value threshold for identifying b0 volumes (default: 40) |

## Pipeline Stages

### Step 1: Input Validation

Scans the input directory for exactly two NIfTI files that each have companion `.json`, `.bval`, and `.bvec` sidecar files. Validates that `TotalReadoutTime` and `PhaseEncodingDirection` are present in the JSON sidecars. Auto-detects which acquisition has the negative phase-encoding direction (assigned as `dtiR`) and which has the positive direction (assigned as `dti`). Copies and stages all files into `<output_dir>/tmp/`, converting uncompressed `.nii` to `.nii.gz` if needed.

### Step 2: Topup + Eddy

Corrects susceptibility-induced and eddy-current distortions:

1. **Odd dimension padding** — If any spatial dimension is odd, zero-pads it by one voxel using numpy so that topup/eddy can operate correctly. Disable with `--no-pad-odd`.
2. **Extract b0 volumes** — Loads both acquisitions once with nibabel, extracts b0 volumes (b-value < `b0tolerance`) by array indexing, and concatenates them into a single 4D file.
3. **Acquisition parameters** — Builds `acq_param.txt` from the phase-encoding directions and total readout time in the JSON sidecars.
4. **Topup** — Estimates the susceptibility-induced off-resonance field from the opposing-PE b0 volumes. Selects a configuration file (`b02b0_4.cnf`, `b02b0.cnf`, or `b02b0_1.cnf`) based on whether dimensions are divisible by 4, 2, or neither.
5. **Brain extraction** — Averages the topup-corrected b0 volumes and runs `bet` to create a brain mask.
6. **Merge volumes** — Concatenates the two full DWI datasets. B-values are rounded to the nearest 100 for eddy compatibility. B-vectors are concatenated and an index file is written mapping each volume to its acquisition parameters line.
7. **Eddy** — Runs eddy (preferring `eddy_cuda` > `eddy_openmp` > `eddy`) with `--repol` (outlier replacement) and `--data_is_shelled` to correct eddy currents, subject motion, and remaining susceptibility distortion.

### Step 3: DTI Fit

Runs FSL `dtifit` on the eddy-corrected data using the rotated b-vectors and brain mask to compute the diffusion tensor at each voxel. Produces standard DTI maps including FA (fractional anisotropy), MD, and eigenvector images. Also creates a thresholded/eroded FA binary mask (`_FA_thr.nii.gz`) at FA > 0.15, used later as a brain mask for bedpostx.

### Step 4: Rim Clean

Removes artifactually bright voxels at the brain boundary of the FA map. Uses a distance transform to identify the outermost shell of non-zero voxels (distance = 1) and their immediate interior neighbours (distance = 2). Any boundary voxel whose FA exceeds the median FA of its interior neighbours is zeroed out. Outputs a cleaned FA image (`r<baseName>_FA.nii.gz`).

### Step 5: MMORF Atlas Registration

Registers the FMRIB58 FA template to the subject's native-space cleaned FA map using a two-stage approach:

1. **FLIRT** — 12-DOF affine registration of the template to the subject FA.
2. **MMORF** — Non-linear (Metal-accelerated) registration using a multi-resolution warp field, refining from 32mm down to high resolution.
3. **Apply warps** — Warps the FMRIB58 template (spline interpolation) and the HarvardOxford cortical atlas (nearest-neighbour interpolation) into subject space.

### Step 6: Bedpostx

*(Skipped with `--no-bedpost`)*

Stages the eddy-corrected data, rotated b-vectors, b-values, and thresholded FA mask into `<output_dir>/bedpost/`. Auto-detects whether `probtrackx2_gpu` supports compressed NIfTI and sets `FSLOUTPUTTYPE` accordingly (defaults to `NIFTI` for uncompressed outputs to avoid gzip decompression overhead during probtrackx). Runs `bedpostx` (preferring GPU version) to estimate fibre orientation distributions at each voxel using a Bayesian multi-fibre model. Outputs are written to `<output_dir>/bedpost.bedpostX/`. This step can take many hours.

### Step 7: Extract Masks

Splits the native-space HarvardOxford atlas into individual binary mask files — one per labelled region — saved as `masks/<index>.nii` in the output directory.

### Step 8: Probtrackx

*(Skipped with `--no-bedpost`)*

Runs probabilistic tractography (`probtrackx2_gpu`) from each atlas region mask as a seed, using the bedpostx fibre orientations. Generates 5000 streamlines per seed voxel. If `probtrackx2_gpu` supports the `--seedlist` option, runs a single invocation with all seeds (loads bedpostx data once). Otherwise falls back to serial invocations via `fsl_sub`.

### Step 9: Fiber Quantify

*(Skipped with `--no-bedpost`)*

Computes pairwise structural connectivity between all atlas regions from the probtrackx output. For each pair of regions (i, j), extracts the streamline density of region i's tractography within region j's mask and vice versa. Produces four symmetric connectivity matrices saved as both `.npy` and `.tsv`:

- **density** — Normalised connection density (fiber count / total possible streamlines)
- **fiber_count** — Raw bidirectional streamline counts
- **mean** — Mean streamline probability across target mask voxels
- **max** — Maximum streamline probability across target mask voxels

## Dependencies

- **Python**: nibabel, numpy, scipy
- **FSL**: eddy (cuda/openmp), topup, flirt, fslmerge, fslmaths, bet, dtifit, applywarp, bedpostx (gpu/cpu), probtrackx2_gpu, fsl_sub
- **Other**: mmorf
- **Environment**: `FSLDIR` must be set for template/atlas lookup

## Incremental Re-runs

All steps check for existing outputs before running (`step_done()`). To skip already-completed stages, simply re-run the pipeline on the same output directory. Use `--force` to re-run all steps regardless.
