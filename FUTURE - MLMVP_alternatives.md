Two viable setups for strong middle-of-video analysis
Option A: Separate JSON artifacts per duration bucket with separate training harnesses

What it is
Each duration bucket writes its own train-ready JSON or Parquet with hook window, middle features, and closing window. Each bucket has its own training script.

Pros
• Simple ingestion for training because each job reads a single small artifact
• Bucket specific features can evolve quickly without touching other buckets
• Clear experiment freezing because artifacts are immutable per bucket
• Compute efficient because there is no filtering or slicing at run time

Cons
• More upkeep because there are several schemas instead of one
• Risk of drift between buckets if you change features often
• Monitoring and drift checks live per bucket, so you keep several dashboards
• Backfills and refactors require repeated work across buckets

How to execute this path well
• One feature catalog for all buckets in a single YAML with fields name, type, unit, compute function, buckets enabled
• Code generated schemas per bucket from that catalog so you never hand edit schemas
• One shared features library for window math, binning, peak detection, slopes, and rhythm
• One validator script that picks the right schema by bucket and model and runs in CI
• Additive change policy where new columns are allowed and removals require a version bump for that bucket
• Version stamps in every row including schema version, generator version, and extracted at
• Run manifest per experiment that lists artifact paths, code commit, and scaler versions
• Minimal common core in every bucket so tooling can be shared later
early_density, mid_density, late_density, middle_len_sec, bins_present
• Storage layout by model and bucket
artifacts or model equals rf or bucket equals 31 to 60 s or date equals YYYYMMDD or rf_31_to_60s_train.parquet

Middle features that restore temporal insight
• Inside the middle only
quartile densities or thirds depending on duration, key shape stats such as peak value and peak position normalized inside the middle, oscillations with a simple prominence threshold, trend slope, variance, coefficient of variation
• Optional for longer videos
piecewise linear fit of the middle with two break points and three slopes, rhythm features such as burstiness of inter event intervals and spectral centroid of event rate
• Always keep hook and closing as fixed three second windows with no overlap into the middle

Option B: One unified JSON per video with middle quartiles inside the middle only

What it is
A single canonical JSON per video contains hook window, middle window, and closing window. The middle window is analyzed with quartiles that live strictly inside the middle. Training jobs materialize per bucket views from this one source.

Pros
• One source of truth avoids drift and keeps governance simple
• One loader, one scaler pattern, and shared CI and monitoring
• Easier backfills and reprocessing because there is a single contract
• Cross bucket comparability is available if you ever need it
• Analytics and diagnostics are faster because every notebook reads the same schema

Cons
• Lowest common denominator risk if some buckets would benefit from richer features than others
• For very short videos the middle might be thin and quartiles may collapse
• Loader code may still materialize per bucket views for training which adds a small ETL step
• Feature changes couple all buckets, so you plan changes a bit more carefully

Mitigations that keep it flexible
• Keep a small stable core plus optional namespaced sections per duration
core fields are always present while sections such as middle extras for longer videos appear only when available
• Use masks and null rules so the schema stays fixed even when bins are skipped for short middles
• Partition the canonical dataset by duration bucket and date to keep IO small
• For training, publish per bucket Parquet views so the trainers still read one small file each
• Snapshot training runs with a manifest so reproducibility remains easy

Middle features that restore temporal insight
• Inside the middle only
q1 to q4 densities, peak value and peak position normalized inside the middle, oscillations, trend slope, variance, coefficient of variation
• Optional extras for longer videos
two break piecewise fit and rhythm features as above
• Hook and closing remain fixed three second windows and never overlap with the middle

Which to choose for your product goal

Your goal is to recover narrative arc, critical moments, peaks, and rhythm inside the middle. That signal comes from the features above, not from the storage choice.

• Choose separate per bucket artifacts if you value speed of iteration, you are comfortable maintaining bucket specific code, and your monitoring will be bucket by bucket
• Choose one unified JSON with middle quartiles inside the middle if you want the lowest maintenance over time, a single loader and validator, and easy backfills while still publishing small per bucket training views

Either path fully supports strong temporal features in the middle as long as you keep all middle bins and shape stats strictly inside the middle window and keep hook and closing as fixed three second windows with no overlap.