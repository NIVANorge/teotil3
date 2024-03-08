# TEOTIL3 development notebooks

Data processing and intermediate analysis for developing TEOTIL3. For a detailed description of proposed development tasks, see the [forprosjekt report](https://niva.brage.unit.no/niva-xmlui/handle/11250/2985726). Tasks listed below have been prioritised for development by Miljødirektoratet and undertaken by NIVA. Task numbers correspond to those in the forprosjekt report.

### Task 2.1: Update core datasets

 * [Notebook 2.1a: Processing historic administrative boundaries](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-1a_historic_admin_boundaries.ipynb)
 * [Notebook 2.1b: Summarise and combine regine-level data](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-1b_update_core_datasets.ipynb)
 * [Notebook 2.1c: Transfer new datasets to PostGIS](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-1c_create_postgis.ipynb)
 * [Notebook 2.1d: Annual data processing/updating](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-1d_annual_data_upload.ipynb)
 * [Notebook 2.1e: Compare databases for TEOTIL2 and TEOTIL3](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-1e_compare_oracle_postgis.ipynb.ipynb)
 
### Task 2.2: Generate catchment hierarchy

 * [Notebook 2.2: Generate catchment hierarchy](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-2_catchment_hierarchy.ipynb)

### Task 2.4: Estimate residence times

 * [Notebook 2.4a: Estimate lake volumes](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-4a_estimate_lake_volumes.ipynb)
 * [Notebook 2.4b: Estimate flow rates](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-4b_estimate_residence_times.ipynb)
 
### Task 2.5: Estimating retention

 * [Notebook 2.5a: Estimating retention parameters](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-5a_est_vollenweider_params_from_data.ipynb)
 * [Notebook 2.5b: Retention for regine catchments](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-5b_retention_coefficients.ipynb)
 * [Notebook 2.5c: Exploring transmission of nutrients to the coast/border](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-5c_transmission_to_coast.ipynb)
 * [Notebook 2.5d: Retention estimates for Mjøsa](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-5d_mjosa_retention.ipynb)

### Task 2.6: Improve agricultural workflow

 * [Notebook 2.6: Incorporating updated agricultural data from NIBIO](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-6_improve_agri_workflow.ipynb)
   
### Task 2.7: Improve workflow for non-agricultural diffuse inputs

 * [Notebook 2.7a: The "1000 Lakes" dataset](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-7a_non_agri_diffuse_inputs.ipynb)
 * [Notebook 2.7b: Spatial interpolation for woodland and upland areas](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-7b_spatial_interpolation_skog_fjell.ipynb)
 * [Notebook 2.7c: Inputs from urban, glacial and lake areas](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-7c_other_background_coeffs.ipynb)

### Task 2.8: Improve workflow for wastewater treatment and industry

 * [Notebook 2.8: Wastewater and industry](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-8_improve_wastewater_industry_workflow.ipynb)
 
### Task 2.9: Improve workflow for aquaculture

 * [Notebook 2.9: Aquaculture](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-9_improve_aquaculture_workflow.ipynb)

### Task 2.11: Additional input sources

 * [Notebook 2.11a: Detailed data for small wastewater sites](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-11a_detailed_data_for_small_wastewater.ipynb)

### Task 2.12: Historic data series

 * [Notebook 2.12a: Restructure historic series](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-12a_restructure_historic_data.ipynb)
 * [Notebook 2.12b: Quality check outlet locations](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-12b_check_outlet_locs.ipynb)
 
### Task 2.15: Testing, documentation and reporting

 * [Notebook 2.15a: Compare TEOTIL2 vs TEOTIL3](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-15a_explore_input_output.ipynb)
 * [Notebook 2.15b: Compare TEOTIL3 against observed data](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-15b_compare_measured_fluxes.ipynb)
 * [Notebook 2.15c: Explore TEOTIL3 input data](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-15c_explore_input_data.ipynb)
 * [Notebook 2.15d: Calibrate retention](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-15d_calibrate_retention.ipynb)
 * [Notebook 2.15e: Case study: Mjøsa](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-15e_mjosa.ipynb)
 * [Notebook 2.15f: Case study: Hålandsvatnet](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-15f_halandsvatnet.ipynb)
 * [Notebook 2.15g: Case study: Vansjø](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-15g_vansjo.ipynb)
 * [Notebook 2.15h: Case study: Vikedal](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-15h_vikedal.ipynb)
 * [Notebook 2.15i: Case study: Referanseelver](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-15i_referanseelver.ipynb)

### Task 2.16: Historic series for TOTN and TOTP

 * [Notebook 2.16: Bias-correcting TEOTIL2 output using TEOTIL3](https://nbviewer.org/github/NIVANorge/teotil3/blob/main/notebooks/development/T2-16_teotil2_vs_teotil3_timeseries.ipynb)

