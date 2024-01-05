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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Encoder(str, Enum):
    cjxl = "cjxl"
    cwp2 = "cwp2"
    flif = "flif"

EFFORT_RANGES = {
    Encoder.cjxl: range(1, 10),
    Encoder.cwp2: range(0, 10),
    Encoder.flif: range(0, 101),
}

app = typer.Typer()

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

def run_compression(file, output_path, compression_function, format_name, effort_range, **kwargs) -> (int, int, float):
    results = []

    for effort in effort_range:
        print(file, effort)
        start_time = time.time()
        result = compression_function(file, f"{output_path}_{effort}", effort=effort, **kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time
        compressed_file_size = os.path.getsize(f"{output_path}_{effort}.{format_name}")
        original_file_size = os.path.getsize(file)

        results.append((os.path.basename(file), original_file_size, compressed_file_size, elapsed_time, effort))

    return results 


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
def plot(file):
    df = pd.read_csv(file)

    # Plot for Compression Ratio
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    ax1 = plt.axes()
    boxplot1 = sns.boxplot(x='Effort', y='Compression Ratio', data=df, ax=ax1)
    boxplot1.set(xlabel='Effort', ylabel='Compression Ratio', title='Compression Ratio vs. Effort')

    for i, group in enumerate(df.groupby('Effort')):
        effort, effort_data = group
        max_index = effort_data['Compression Ratio'].idxmax()
        filename_max = df.loc[max_index, 'Filename']
        ax1.text(i, effort_data['Compression Ratio'].max(), f'Max: {filename_max}', color='red', weight='bold',
                 ha='center', va='center', fontsize=8)
        
        min_index = effort_data['Compression Ratio'].idxmin()
        filename_min = df.loc[min_index, 'Filename']
        ax1.text(i, effort_data['Compression Ratio'].min(), f'Min: {filename_min}', color='red', weight='bold',
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
        ax2.text(i, effort_data['Time Elapsed (seconds)'].max(), f'Max: {filename_max}', color='red', weight='bold',
                 ha='center', va='center', fontsize=8)
        
        min_index = effort_data['Time Elapsed (seconds)'].idxmin()
        filename_min = df.loc[min_index, 'Filename']
        ax2.text(i, effort_data['Time Elapsed (seconds)'].min(), f'Min: {filename_min}', color='red', weight='bold',
                 ha='center', va='center', fontsize=8)

    plt.show()

@app.command()
def run(encoder: Encoder = typer.Option(
        None,
        "--encoder",
        help="Specify an encoder. Use this option if you want to compress using only one encoder.",
        show_choices=True,
    ), input_dir: Path = typer.Argument(
        os.getcwd(),
        help="Specify path to input directory",
    ), output_dir: Path = typer.Argument(
        os.getcwd(),
        help="Specify path to output directory",
    )):

    jxl_results = []
    wp_results = []
    flif_results = []

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for file in files:
        output_path = os.path.join(output_dir, os.path.basename(file).split('.')[0])

        if (encoder == Encoder.cjxl):
            jxl_results.extend(run_compression(file, output_path, compress_with_jxl, "jxl", range(1, 10)))
        elif (encoder == Encoder.cwp2):
            wp_results.extend(run_compression(file, output_path, compress_with_webp, "wp", range(10)))
        elif (encoder == Encoder.flif):
            flif_results.extend(run_compression(file, output_path, compress_with_flif, "flif", range(101)))
        else:   
            jxl_results.extend(run_compression(file, output_path, compress_with_jxl, "jxl", range(1, 10)))
            wp_results.extend(run_compression(file, output_path, compress_with_webp, "wp", range(10)))
            flif_results.extend(run_compression(file, output_path, compress_with_flif, "flif", range(101)))

    if (jxl_results):
        write_to_csv(jxl_results, 'compression_results_jxl.csv')
    if (wp_results):
        write_to_csv(wp_results, 'compression_results_wp.csv')
    if (flif_results):
        write_to_csv(flif_results, 'compression_results_flif.csv')

if __name__ == '__main__':
    app()
