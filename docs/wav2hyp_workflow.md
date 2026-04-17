# WAV2HYP Workflow

Waveform-to-hypocenter pipeline: data inputs, processing steps, and outputs.

## Main pipeline flowchart

```mermaid
flowchart LR
    subgraph inputs["Data inputs"]
        config["config.yaml"]
        target["Target location\nlat, lon, elev"]
        inv["Station inventory\n(StationXML)"]
        waves["Waveform data\n(FDSN / SDS / Earthworm / SeedLink)"]
        trange["Time range\n(--t1, --t2)"]
    end

    subgraph cli["Entry"]
        cmd["wav2hyp CLI\n(cli.py)"]
    end

    subgraph core["Core"]
        run["WAV2HYP.run()\n(core.py)"]
    end

    subgraph steps["Processing steps"]
        pick["1. Picker\n_run_picker()"]
        assoc["2. Associator\n_run_associator()"]
        loc["3. Locator\n_run_locator()"]
    end

    subgraph libs["Libraries / codes"]
        vclient["VClient\n(vdapseisutils)"]
        eqt["EQTransformer\n(seisbench)"]
        pyocto["PyOcto"]
        nll["NonLinLoc\n(nllpy)"]
    end

    subgraph outputs["Outputs"]
        out_pick["picks/eqt-volpick.h5"]
        out_assoc["associations/pyocto.h5"]
        out_loc["locations/nll.h5\n+ .hyp files"]
        summary["Summary .txt"]
        catalog["VCatalog"]
        logs["wav2hyp.log"]
    end

    target --> config
    config --> cmd
    inv --> cmd
    trange --> cmd
    cmd --> run
    run --> pick
    waves --> vclient
    vclient --> pick
    inv --> pick
    pick --> eqt
    eqt --> out_pick
    pick --> assoc
    out_pick -.-> assoc
    inv --> assoc
    assoc --> pyocto
    pyocto --> out_assoc
    assoc --> loc
    out_assoc -.-> loc
    inv --> loc
    loc --> nll
    nll --> out_loc
    out_loc --> catalog
    out_pick --> summary
    out_assoc --> summary
    out_loc --> summary
    run --> logs
```

**Important:** The **target location** (latitude, longitude, elevation) is a key input to `config.yaml` (under `target:`). It defines the study area for association (PyOcto) and the grid origin for location (NonLinLoc via nllpy).

## Inputs to NonLinLoc (parallel view)

NonLinLoc is driven by [nllpy](https://github.com/jwellik/nllpy), which builds the control file and runs Vel2Grid → Grid2Time → NLLoc. All of the following feed into NonLinLoc:

```mermaid
flowchart TB
    subgraph config_inputs["From config.yaml"]
        target_nll["Target location\nlat, lon, elev"]
        nll_home["nll_home\n(working directory)"]
        config_name["config_name\nstation_format"]
    end

    subgraph from_pipeline["From pipeline"]
        obs["Observation files\n(NLLOC_OBS)\nfrom associated catalog"]
        inv_nll["Station inventory\n→ GTSRCE / EQSTA"]
    end

    subgraph nllpy_build["nllpy builds"]
        trans["TRANS\nlat_orig, lon_orig\n(coordinate origin)"]
        layer["Velocity model\nlayer.layers\n(P, S by depth)"]
        vggrid["Velocity grid\nvggrid\n(size, spacing)"]
        locgrid["Location grid\nlocgrid\n(search volume)"]
        locsearch["LOCSEARCH\nsearch type, cells"]
        locmethod["LOCMETHOD\nmin phases, weights"]
        ctrl["Control file\n.in"]
    end

    subgraph nll_run["NonLinLoc run"]
        vel2grid["Vel2Grid"]
        grid2time["Grid2Time"]
        nlloc["NLLoc"]
    end

    target_nll --> trans
    target_nll --> locgrid
    nll_home --> ctrl
    config_name --> ctrl
    obs --> ctrl
    inv_nll --> ctrl
    trans --> ctrl
    layer --> ctrl
    vggrid --> ctrl
    locgrid --> ctrl
    locsearch --> ctrl
    locmethod --> ctrl
    ctrl --> vel2grid
    vel2grid --> grid2time
    grid2time --> nlloc
```

| Input to NonLinLoc | Source | Role |
|--------------------|--------|------|
| **Target location (lat, lon, elev)** | `config.target` | Grid origin in nllpy (`create_volcano_config(lat_orig, lon_orig)`); defines TRANS and grid geometry. |
| **Observation files (NLLOC_OBS)** | Associated catalog | Picks per event; written by `catalog.write_nlloc_obs()`. |
| **Station inventory** | `config.inventory` | Station coordinates and (optionally) phase errors → GTSRCE/EQSTA in control file. |
| **nll_home** | `config.locator.nll_home` | Working directory for control file, grids, and output. |
| **Velocity model** | nllpy template (e.g. volcano) | `config.layer.layers` (depth, Vp, Vs, etc.). |
| **Velocity / location grids** | nllpy template | vggrid, locgrid (size, spacing, origin). |
| **Control file** | nllpy `write_complete_control_file()` | Ties all of the above together; Vel2Grid, Grid2Time, and NLLoc read it. |

## Detailed workflow (left-to-right)

| Stage | Inputs | Code / component | Outputs |
|--------|--------|-------------------|--------|
| **Entry** | `config.yaml`, `--t1`/`--t2`, `-p`/`-a`/`-l`/`--all` | `wav2hyp.cli` → `WAV2HYP(config).run()` | Time chunks, validated config |
| **1. Picker** | Inventory, time range, waveform client config | `VClient` (waveforms), `seisbench.EQTransformer` (annotate), `EQTOutput` | `picks/eqt-volpick.h5`, optional picker summary |
| **2. Associator** | Picks (from step 1 or existing .h5), inventory, associator config | `pyocto.OctoAssociator`, `PyOctoOutput` | `associations/pyocto.h5`, optional associator summary |
| **3. Locator** | Associated catalog (from step 2 or existing .h5), inventory, NLL config | `nllpy` (control file), NonLinLoc (vel2grid, grid2time, NLLoc), `NLLOutput` | `locations/nll.h5`, NLL `.hyp` files, optional locator summary |
| **End** | Located catalog | — | Final `VCatalog`, logs |

## Input summary

- **config.yaml**: **Target location** (`target`: name, **latitude**, **longitude**, **elevation**) is a key input—used for association ROI and for NonLinLoc grid origin via nllpy. Also: `inventory.file`, `waveform_client` (datasource, client_type), `output` (base_dir, dirs, summary filenames), `picker`, `associator`, `locator` sections.
- **Station inventory**: path in config; StationXML format.
- **Waveform data**: accessed via VClient using config (FDSN, SDS path, Earthworm, or SeedLink).
- **Time range**: required for processing; `--t1` and `--t2`.

## Velocity model and NLLPy in config

**Can the user define the velocity model in wav2hyp's config?**  
Yes, in two ways:

1. **`locator.velocity_model_layers`** — Optional. List of layer rows for NonLinLoc. Each row is `[depth_km, VpTop, VpGrad, VsTop, VsGrad, rhoTop, rhoGrad]`. If present, this overrides the default velocity model from nllpy’s volcano template.
2. **`locator.nllpy_overrides`** — Optional. Dict of options passed through to the nllpy config after the volcano template is applied. Use nested keys for sub-objects (e.g. `locgrid: { d_grid_x: 0.2, d_grid_y: 0.2 }`, or `layer: { layers: [...] }`). This lets you change grids, search, location method, phase IDs, etc., without changing nllpy code.

**Associator velocity** (used only for PyOcto association, not for NonLinLoc) is set in `associator.p_velocity` and `associator.s_velocity`.

**Can the user define inputs for NLLPy?**  
Yes. wav2hyp passes to nllpy:

- From config: **target** (lat, lon) → `create_volcano_config(lat_orig, lon_orig)`, plus `nll_home`, `config_name`, `station_format`, `run_vel2grid`, `run_grid2time`.
- Optional: **`velocity_model_layers`** and **`nllpy_overrides`** (see above). Any other nllpy/NLLoc settings can be overridden via `nllpy_overrides` if the corresponding config attribute exists on nllpy’s `NLLocConfig` (e.g. `locgrid`, `vggrid`, `locsearch`, `locmethod`, `layer`).

## Output summary

- **Picker**: `{base_dir}/{picker_dir}/eqt-volpick.h5` (picks + detections); optional `picker_summary` .txt.
- **Associator**: `{base_dir}/{associator_dir}/pyocto.h5` (events + assignments); optional `associator_summary` .txt.
- **Locator**: `{base_dir}/{locator_dir}/nll.h5` (catalog), plus NonLinLoc `.hyp` under `nll_dir`; optional `locator_summary` .txt.
- **Logs**: `{base_dir}/{log_dir}/wav2hyp.log`.
- **In-memory**: final `vdapseisutils.VCatalog` returned by `run()`.
