import os
import subprocess
import sys
import shutil
import time
import uuid
import csv
import typer
import cv2
from enum import Enum
from pathlib import Path
from typing import Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor


class Encoder(str, Enum):
    cjxl = "cjxl"
    cwp2 = "cwp2"
    flif = "flif"

EFFORT_RANGES = {
    Encoder.cjxl: {'range': range(1, 10), 'step': 1},
    Encoder.cwp2: {'range': range(0, 10), 'step': 1},
    Encoder.flif: {'range': range(0, 101), 'step': 10},
}

app = typer.Typer()

def compress_with_jxl(file, output_path, effort=7):
    result = subprocess.run(["./cjxl", "-d", "0", "-e", str(effort), file, output_path + ".jxl"], capture_output=True, encoding="utf-8")
    return result

def compress_with_webp(file, output_path, effort=5):
    result = subprocess.run(["./cwp2", "-q", "100", "-effort", str(effort), file, "-o", output_path + ".wp2"], capture_output=True, encoding="utf-8")
    return result

def compress_with_flif(file, output_path, effort=60):
    result = subprocess.run(["./flif", "-E", str(effort), file,  output_path + ".flif"], capture_output=True, encoding="utf-8")
    return result
    
def timed(func):
    def _w(*a, **k):
        then = time.time()
        res = func(*a, **k)
        elapsed = time.time() - then
        return elapsed, res
    return _w

def compression_task(file, output_path, compression_function, format_name, effort):
    return timed(compression_function)(file, f"{output_path}_{effort}", effort)

def run_compression(file, output_path, compression_function, format_name, effort_range, **kwargs) -> list:
    results = []

    im = cv2.imread(file)
    h, w, _ = im.shape
    original_size = 3 * h * w

    with ProcessPoolExecutor(max_workers=10) as executor:
        tasks = []

        for effort in effort_range:
            task = executor.submit(compression_task, file, output_path, compression_function, format_name, effort)
            tasks.append(task)

        for task, effort in zip(tasks, effort_range):
            elapsed, result = task.result()

            compressed_file_size = os.path.getsize(f"{output_path}_{effort}.{format_name}")
            compression_ratio = original_size / compressed_file_size
            perc_of_original = compressed_file_size / original_size

            results.append((os.path.basename(file), original_size, w, h, compressed_file_size, compression_ratio, perc_of_original, elapsed, effort))

    return results

def write_to_csv(results, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Filename', 'Original File Size (bytes)', 'Width', 'Height', 'Compressed File Size (bytes)', 'Compression Ratio', '% of Original', 'Time Elapsed (seconds)', 'Effort']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for filename, original_size, w, h, compressed_size, compression_ratio, perc_of_original, elapsed_time, effort in results:
            writer.writerow({
                'Filename': filename,
                'Original File Size (bytes)': original_size, 
                'Width': w,
                'Height': h,
                'Compressed File Size (bytes)': compressed_size,
                'Compression Ratio': compression_ratio,
                '% of Original': perc_of_original,
                'Time Elapsed (seconds)': elapsed_time,
                'Effort': effort
            })

@app.command()
def plot(file):
    df = pd.read_csv(file)

    # Plot for Compression Ratio
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    ax1 = plt.axes()
    boxplot1 = sns.boxplot(x='Effort', y='% of Original', data=df, ax=ax1)
    boxplot1.set(xlabel='Effort', ylabel='% of Original', title='% of Original vs. Effort')

    for i, group in enumerate(df.groupby('Effort')):
        effort, effort_data = group
        max_index = effort_data['% of Original'].idxmax()
        filename_max = df.loc[max_index, 'Filename']
        ax1.text(i, effort_data['% of Original'].max(), f'{filename_max}', color='red', weight='bold',
                 ha='center', va='center', fontsize=8)
        
        min_index = effort_data['% of Original'].idxmin()
        filename_min = df.loc[min_index, 'Filename']
        ax1.text(i, effort_data['% of Original'].min(), f'{filename_min}', color='red', weight='bold',
                 ha='center', va='center', fontsize=8)

    plt.show()

    # Plot for Time Elapsed
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    ax2 = plt.axes()
    boxplot2 = sns.boxplot(x='Effort', y='Time Elapsed (seconds)', data=df, ax=ax2)
    boxplot2.set(xlabel='Effort', ylabel='Time Elapsed (seconds)', title='Time Elapsed vs. Effort')

    ax2.set_yscale('log')

    for i, group in enumerate(df.groupby('Effort')):
        effort, effort_data = group
        max_index = effort_data['Time Elapsed (seconds)'].idxmax()
        filename_max = df.loc[max_index, 'Filename']
        ax2.text(i, effort_data['Time Elapsed (seconds)'].max(), f'{filename_max}', color='red', weight='bold',
                 ha='center', va='center', fontsize=8)
        
        min_index = effort_data['Time Elapsed (seconds)'].idxmin()
        filename_min = df.loc[min_index, 'Filename']
        ax2.text(i, effort_data['Time Elapsed (seconds)'].min(), f'{filename_min}', color='red', weight='bold',
                 ha='center', va='center', fontsize=8)

    plt.show()

@app.command()
def run(encoder: Encoder = typer.Option(
        None,
        "--encoder",
        help="Specify an encoder. Use this option if you want to compress using only one encoder.",
        show_choices=True,
    ), compare: bool = typer.Option(
        False,
        "--compare",
        help="Compare decompressed image to original image"
    ), input_dir: Path = typer.Argument(
        os.getcwd(),
        help="Specify path to input directory",
    ), output_dir: Path = typer.Argument(
        os.getcwd(),
        help="Specify path to output directory",
    ), output_csv: str = typer.Argument(
        "compression_results",
        help="Specify name of output csv file",
    )):

    jxl_results = []
    wp_results = []
    flif_results = []

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    start_time_total = time.time()

    for file in files:
        output_path = os.path.join(output_dir, os.path.basename(file).split('.')[0])

        if (encoder == Encoder.cjxl):
            jxl_results.extend(run_compression(file, output_path, compress_with_jxl, "jxl", range(1, 10)))
        elif (encoder == Encoder.cwp2):
            wp_results.extend(run_compression(file, output_path, compress_with_webp, "wp2", range(10)))
        elif (encoder == Encoder.flif):
            flif_results.extend(run_compression(file, output_path, compress_with_flif, "flif", range(0, 101, 10)))
        else:   
            jxl_results.extend(run_compression(file, output_path, compress_with_jxl, "jxl", range(1, 10)))
            wp_results.extend(run_compression(file, output_path, compress_with_webp, "wp2", range(10)))
            flif_results.extend(run_compression(file, output_path, compress_with_flif, "flif", range(0, 101, 10)))

    end_time_total = time.time()  
    total_elapsed_time = end_time_total - start_time_total
    print(f"\nTotal Time Elapsed for compression: {total_elapsed_time:.4f} seconds")
    
    if jxl_results:
        write_to_csv(jxl_results, output_csv + '_jxl.csv')
        avg_jxl = calculate_average_values(jxl_results)
        print(f"\nAverage Compression Ratio for JPEG XL: {avg_jxl[0]:.4f}")
        print(f"Average Time Elapsed (seconds) for JPEG XL: {avg_jxl[1]:.4f}")
    if (wp_results):
        write_to_csv(wp_results,  output_csv + '_wp2.csv')
        avg_wp = calculate_average_values(wp_results)
        print(f"\nAverage Compression Ratio for WebP: {avg_wp[0]:.4f}")
        print(f"Average Time Elapsed (seconds) for WebP: {avg_wp[1]:.4f}")
    if (flif_results):
        write_to_csv(flif_results, output_csv + '_flif.csv')
        avg_flif = calculate_average_values(flif_results)
        print(f"\nAverage Compression Ratio for Flif: {avg_flif[0]:.4f}")
        print(f"Average Time Elapsed (seconds) for Flif: {avg_flif[1]:.4f}")
   

def calculate_average_values(results):
    if not results:
        return 0, 0

    total_compression_ratio = 0
    total_elapsed_time = 0

    for _, original_size, _, _, compressed_size, compression_ratio, _, elapsed_time, _ in results:
        total_compression_ratio += compression_ratio
        total_elapsed_time += elapsed_time

    average_compression_ratio = total_compression_ratio / len(results)
    average_elapsed_time = total_elapsed_time / len(results)

    return average_compression_ratio, average_elapsed_time

if __name__ == '__main__':
    app()
