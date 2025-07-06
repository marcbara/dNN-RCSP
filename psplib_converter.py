#!/usr/bin/env python3
"""
PSPLIB to JSON Converter
========================

Converts PSPLIB format files (.sm) to JSON format compatible with DatalessProjectScheduler.
PSPLIB is the standard library for project scheduling problems used in academic research.

Usage:
    python psplib_converter.py input.sm output.json
    python psplib_converter.py --batch input_folder/ output_folder/

Author: Marc Bara
"""

import json
import re
import os
import argparse
from pathlib import Path

def parse_psplib_file(filepath):
    """
    Parse a PSPLIB .sm file and extract project data.
    
    Args:
        filepath: Path to the PSPLIB .sm file
        
    Returns:
        dict: Parsed project data with durations, precedences, and metadata
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract basic project information
    jobs_match = re.search(r'jobs \(incl\. supersource/sink \):\s*(\d+)', content)
    if not jobs_match:
        raise ValueError(f"Could not find job count in {filepath}")
    
    n_jobs = int(jobs_match.group(1))
    
    # Extract precedence relations
    precedence_section = re.search(r'PRECEDENCE RELATIONS:.*?jobnr\.\s+#modes\s+#successors\s+successors\s*\n(.*?)\n\*+', 
                                 content, re.DOTALL)
    if not precedence_section:
        raise ValueError(f"Could not find precedence relations in {filepath}")
    
    precedence_lines = precedence_section.group(1).strip().split('\n')
    
    # Parse precedence relations
    successors = {}
    for line in precedence_lines:
        line = line.strip()
        if not line or line.startswith('*'):
            continue
            
        parts = line.split()
        if len(parts) < 3:
            continue
            
        job_nr = int(parts[0])
        n_successors = int(parts[2])
        
        if n_successors > 0 and len(parts) >= 3 + n_successors:
            successors[job_nr] = [int(parts[3 + i]) for i in range(n_successors)]
        else:
            successors[job_nr] = []
    
    # Extract durations
    duration_section = re.search(r'REQUESTS/DURATIONS:.*?jobnr\.\s+mode\s+duration.*?\n(.*?)\n\*+', 
                                content, re.DOTALL)
    if not duration_section:
        raise ValueError(f"Could not find durations in {filepath}")
    
    duration_lines = duration_section.group(1).strip().split('\n')
    
    # Parse durations
    durations = {}
    for line in duration_lines:
        line = line.strip()
        if not line or line.startswith('*'):
            continue
            
        parts = line.split()
        if len(parts) < 3:
            continue
            
        job_nr = int(parts[0])
        duration = int(parts[2])
        durations[job_nr] = duration
    
    return {
        'n_jobs': n_jobs,
        'successors': successors,
        'durations': durations
    }

def convert_to_scheduler_format(psplib_data, instance_name="PSPLIB_Instance"):
    """
    Convert parsed PSPLIB data to DatalessProjectScheduler JSON format.
    
    Args:
        psplib_data: Parsed PSPLIB data
        instance_name: Name prefix for activities
        
    Returns:
        dict: JSON format compatible with DatalessProjectScheduler
    """
    n_jobs = psplib_data['n_jobs']
    successors = psplib_data['successors']
    durations_dict = psplib_data['durations']
    
    # Convert to 0-indexed and create duration array
    durations = []
    for job_nr in range(1, n_jobs + 1):
        if job_nr in durations_dict:
            durations.append(durations_dict[job_nr])
        else:
            durations.append(0)  # Default for missing jobs
    
    # Generate activity names
    names = []
    for i in range(n_jobs):
        if i == 0:
            names.append("START")
        elif i == n_jobs - 1:
            names.append("END")
        else:
            names.append(f"Activity_{i}")
    
    # Convert successors to precedence pairs (predecessor, successor)
    precedences = []
    for job_nr, succ_list in successors.items():
        if job_nr <= n_jobs and succ_list:  # Valid job with successors
            pred_idx = job_nr - 1  # Convert to 0-indexed
            for succ_job in succ_list:
                if succ_job <= n_jobs:  # Valid successor
                    succ_idx = succ_job - 1  # Convert to 0-indexed
                    precedences.append([pred_idx, succ_idx])
    
    return {
        "durations": durations,
        "names": names,
        "precedences": precedences,
        "metadata": {
            "source": "PSPLIB",
            "instance": instance_name,
            "n_jobs": n_jobs,
            "n_precedences": len(precedences)
        }
    }

def convert_psplib_to_json(input_file, output_file):
    """
    Convert a single PSPLIB file to JSON format.
    
    Args:
        input_file: Path to input PSPLIB .sm file
        output_file: Path to output JSON file
    """
    try:
        # Parse PSPLIB file
        psplib_data = parse_psplib_file(input_file)
        
        # Convert to scheduler format
        instance_name = Path(input_file).stem
        scheduler_data = convert_to_scheduler_format(psplib_data, instance_name)
        
        # Write JSON file
        with open(output_file, 'w') as f:
            json.dump(scheduler_data, f, indent=2)
        
        print(f"✅ Converted {input_file} → {output_file}")
        print(f"   📊 {scheduler_data['metadata']['n_jobs']} jobs, {scheduler_data['metadata']['n_precedences']} precedences")
        
    except Exception as e:
        print(f"❌ Error converting {input_file}: {e}")

def batch_convert(input_folder, output_folder):
    """
    Convert all PSPLIB files in a folder.
    
    Args:
        input_folder: Path to folder containing .sm files
        output_folder: Path to output folder for JSON files
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .sm files
    sm_files = list(input_path.glob("*.sm"))
    
    if not sm_files:
        print(f"⚠️  No .sm files found in {input_folder}")
        return
    
    print(f"🔄 Converting {len(sm_files)} PSPLIB files...")
    
    success_count = 0
    for sm_file in sm_files:
        json_file = output_path / f"{sm_file.stem}.json"
        try:
            convert_psplib_to_json(sm_file, json_file)
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to convert {sm_file}: {e}")
    
    print(f"\n🎉 Successfully converted {success_count}/{len(sm_files)} files!")

def main():
    parser = argparse.ArgumentParser(description="Convert PSPLIB files to DatalessProjectScheduler JSON format")
    parser.add_argument("input", help="Input PSPLIB .sm file or folder")
    parser.add_argument("output", help="Output JSON file or folder")
    parser.add_argument("--batch", action="store_true", help="Batch convert all .sm files in input folder")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_convert(args.input, args.output)
    else:
        convert_psplib_to_json(args.input, args.output)

if __name__ == "__main__":
    main() 