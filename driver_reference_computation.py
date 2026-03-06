import sys
import subprocess
from pathlib import Path

def main():

    root = Path("sample_dataset")
    logs_root = Path("logs_dataset")
    model_dir = root / "model"
    gcode_dir = root / "gcode"
    config_dir = root / "config"


    compute_script = Path("compute_time_cost_references_sobol_sampling.py")

    if not compute_script.exists():
        raise FileNotFoundError(
            f"Cannot find {compute_script}. "
            f"Place this driver next to compute_time_cost_references_placeholder_quality.py "
            f"or update the path."
        )


    if not model_dir.is_dir():
        raise FileNotFoundError(f"Expected model directory not found: {model_dir}")
    if not gcode_dir.is_dir():
        raise FileNotFoundError(f"Expected gcode directory not found: {gcode_dir}")
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Expected config directory not found: {config_dir}")


    for stl_path in sorted(model_dir.glob("*.stl")):
        stem = stl_path.stem                                

        ini_path = config_dir / f"{stem}.ini"
        gcode_path = gcode_dir / f"{stem}.gcode"

        if not ini_path.exists():
            print(f"[skip] {stem}: missing config {ini_path}")
            continue

        if not gcode_path.exists():
                                                                   
                                            
            print(f"[warn] {stem}: missing gcode {gcode_path}; continuing anyway")

        output_root_dir = logs_root / f"logs_{stem}"
        output_root_dir.mkdir(parents=True, exist_ok=True)


        cmd = [
            sys.executable,
            str(compute_script),
            "--model_stl", str(stl_path.resolve()),
            "--profile_ini", str(ini_path.resolve()),
            "--output_root_dir", str(output_root_dir.resolve()),
            "--refs_store", str(output_root_dir.resolve())+'/refs_store.json',
            "--seed", "2021",
        ]

        print(f"[run] {stem}: output_root_dir={output_root_dir}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
                                                                                          
            print(f"[error] compute_script failed for {stem} with return code {e.returncode}")

if __name__ == "__main__":
    main()
