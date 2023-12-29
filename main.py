import os
import subprocess
import sys
import shutil
import time
import uuid
import csv
import typer
from enum import Enum
from pathlib import Path
from typing import Optional

class Algorithm(str, Enum):
    cjxl = "cjxl"
    cwp2 = "cwp2"
    flif = "flif"

EFFORT_RANGES = {
    Algorithm.cjxl: range(1, 10),
    Algorithm.cwp2: range(0, 10),
    Algorithm.flif: range(0, 101),
}

app = typer.Typer()

@app.command()
def hello(name: str):
    print(f"Hello {name}")

@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        print(f"Goodbye Ms. {name}. Have a good day.")
    else:
        print(f"Bye {name}!")

def compress_with_jxl(file, output_path, effort=7):
    result = subprocess.run(["./cjxl", "-d", "0", "-e", str(effort), file, output_path + ".jxl"], capture_output=True, encoding="utf-8")
    return result

def compress_with_webp(file, output_path, effort=5):
    result = subprocess.run(["./cwp2", "-q", "100", "-effort", str(effort), file, "-o", output_path + ".wp2"], capture_output=True, encoding="utf-8")
    print(result.stdout)
    return result

def compress_with_flif(file, output_path, effort=60):
    result = subprocess.run(["./flif", "-E", str(effort), file,  output_path + ".flif"], capture_output=True, encoding="utf-8")
    return result

def run_compression(file, output_path, compression_function, format_name, **kwargs) -> (int, int, float):
    start_time = time.time()
    result = compression_function(file, output_path, **kwargs)
    end_time = time.time()

    elapsed_time = end_time - start_time
    compressed_file_size = os.path.getsize(output_path + "." + format_name)
    original_file_size = os.path.getsize(file)

    return (original_file_size, compressed_file_size, elapsed_time)


def write_to_csv(results, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Filename', 'Original File Size (bytes)', 'Compressed File Size (bytes)', 'Compression Ratio', 'Time Elapsed (seconds)', 'Effort']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for filename, original_size, compressed_size, elapsed_time, effort in results:
            writer.writerow({
                'Filename': filename,
                'Original File Size (bytes)': original_size*3, # in case of PNG since there are 3 channels (RGB)
                'Compressed File Size (bytes)': compressed_size,
                'Compression Ratio': compressed_size / (original_size*3),
                'Time Elapsed (seconds)': elapsed_time,
                'Effort': effort
            })

@app.command()
def visualize(file):
    pass

@app.command()
def run(algorithm: Algorithm = typer.Option(
        None,
        "--algorithm",
        help="Specify a single compression algorithm to use",
        show_choices=True,
    ), input_dir: Path = typer.Argument(
        os.getcwd(),
        help="Specify path to input directory",
    ), output_dir: Path = typer.Argument(
        os.getcwd(),
        help="Specify path to output directory",
    )):

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    results = []

    for file in files:

        output_path = os.path.join(output_dir, os.path.basename(file).split('.')[0]) 

        valid_effort_range = EFFORT_RANGES.get(algorithm, [])

        for effort in range(1, 10):
            result = run_compression(file, output_path+"_"+str(effort), compress_with_jxl, "jxl", effort=effort)
            results.append((os.path.basename(file), *result, effort))
        
        write_to_csv(results, 'compression_results_jxl.csv')
        results = []

        for effort in range(0,10):
            result = run_compression(file, output_path+"_"+str(effort), compress_with_webp, "wp2", effort=effort)
            results.append((os.path.basename(file), *result, effort))

        write_to_csv(results, 'compression_results_wp.csv')
        results = []
        
        # for effort in range(0, 101):
        #     filename = str(uuid.uuid4())
        #     result = run_compression(file, output_path+"_"+str(effort), compress_with_flif, "flif", effort=effort)
        #     results.append((os.path.basename(file), *result, effort))

        # write_to_csv(results, 'compression_results_flif.csv')
if __name__ == '__main__':
    app()
