$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$env:HF_HOME = Join-Path $repo "data\hf_cache"
$env:TORCH_HOME = Join-Path $repo "data\torch_cache"
$env:TEMP = Join-Path $repo "outputs\install_tmp"
$models = @("unet", "deeplabv3plus", "segformer_b0")
$seeds = @(13, 42, 87)
foreach ($model in $models) {
    foreach ($seed in $seeds) {
        $out = Join-Path $repo "outputs\part9_reproduction\$model\seed_$seed"
        if ((Test-Path (Join-Path $out "metrics.json")) -and (Test-Path (Join-Path $out "predictions.png"))) {
            Write-Host "Skipping completed: $model seed $seed"
            continue
        }
        $args = @("run", "--no-capture-output", "-n", "remote-sensing-dl", "python", (Join-Path $PSScriptRoot "train_segmentation.py"), "--model", $model, "--data-root", (Join-Path $repo "data\landcoverai"), "--output", $out, "--epochs", "5", "--train-limit", "900", "--val-limit", "240", "--workers", "2", "--seed", $seed)
        if (Test-Path (Join-Path $out "last.pt")) { $args += "--resume" }
        & conda @args
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $model seed $seed" }
    }
}
conda run -n remote-sensing-dl python (Join-Path $PSScriptRoot "aggregate_reproduction.py") --root (Join-Path $repo "outputs\part9_reproduction")
