#!/usr/bin/env python3
"""
Critical Path Analyzer for PSPLIB .sm files
Computes the critical path duration (makespan) for project scheduling instances
"""

import re
import os
from collections import defaultdict, deque
import argparse


class CriticalPathAnalyzer:
    def __init__(self):
        self.jobs = {}  # job_id -> duration
        self.precedences = defaultdict(list)  # job_id -> [successor_ids]
        self.predecessors = defaultdict(list)  # job_id -> [predecessor_ids]
        
    def parse_sm_file(self, filepath):
        """Parse a PSPLIB .sm file and extract precedence relations and durations"""
        self.jobs = {}
        self.precedences = defaultdict(list)
        self.predecessors = defaultdict(list)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract precedence relations
        precedence_section = re.search(
            r'PRECEDENCE RELATIONS:\s*\n.*?\n(.*?)\n\*+',
            content, re.DOTALL
        )
        
        if precedence_section:
            for line in precedence_section.group(1).strip().split('\n'):
                parts = line.split()
                if len(parts) >= 3:
                    job_id = int(parts[0])
                    num_successors = int(parts[2])
                    if num_successors > 0:
                        successors = [int(x) for x in parts[3:3+num_successors]]
                        self.precedences[job_id] = successors
                        for succ in successors:
                            self.predecessors[succ].append(job_id)
        
        # Extract durations
        duration_section = re.search(
            r'REQUESTS/DURATIONS:\s*\n.*?\n(.*?)\n\*+',
            content, re.DOTALL
        )
        
        if duration_section:
            for line in duration_section.group(1).strip().split('\n'):
                parts = line.split()
                if len(parts) >= 3:
                    job_id = int(parts[0])
                    duration = int(parts[2])
                    self.jobs[job_id] = duration
    
    def compute_critical_path(self):
        """Compute the critical path length using longest path algorithm"""
        if not self.jobs:
            return 0
        
        # Find all jobs
        all_jobs = set(self.jobs.keys())
        
        # Initialize earliest start times
        earliest_start = {}
        
        # Topological sort to process jobs in correct order
        in_degree = defaultdict(int)
        for job in all_jobs:
            in_degree[job] = len(self.predecessors[job])
        
        queue = deque([job for job in all_jobs if in_degree[job] == 0])
        
        # Process jobs in topological order
        while queue:
            current_job = queue.popleft()
            
            # Calculate earliest start time for current job
            if not self.predecessors[current_job]:
                earliest_start[current_job] = 0
            else:
                earliest_start[current_job] = max(
                    earliest_start[pred] + self.jobs[pred] 
                    for pred in self.predecessors[current_job]
                )
            
            # Update successors
            for successor in self.precedences[current_job]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        # Find the job with maximum finish time (earliest_start + duration)
        max_finish_time = 0
        critical_job = None
        
        for job in all_jobs:
            finish_time = earliest_start[job] + self.jobs[job]
            if finish_time > max_finish_time:
                max_finish_time = finish_time
                critical_job = job
        
        return max_finish_time, critical_job, earliest_start
    
    def analyze_file(self, filepath):
        """Analyze a single .sm file and return critical path info"""
        try:
            self.parse_sm_file(filepath)
            cp_length, critical_job, earliest_starts = self.compute_critical_path()
            
            return {
                'file': os.path.basename(filepath),
                'critical_path_length': cp_length,
                'critical_job': critical_job,
                'num_jobs': len(self.jobs),
                'num_precedences': sum(len(succs) for succs in self.precedences.values()),
                'success': True
            }
        except Exception as e:
            return {
                'file': os.path.basename(filepath),
                'error': str(e),
                'success': False
            }


def main():
    parser = argparse.ArgumentParser(description='Compute critical path for PSPLIB .sm files')
    parser.add_argument('input', help='Input .sm file or directory containing .sm files')
    parser.add_argument('--output', '-o', help='Output CSV file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    analyzer = CriticalPathAnalyzer()
    results = []
    
    if os.path.isfile(args.input):
        # Single file
        result = analyzer.analyze_file(args.input)
        results.append(result)
    elif os.path.isdir(args.input):
        # Directory of .sm files
        sm_files = [f for f in os.listdir(args.input) if f.endswith('.sm')]
        sm_files.sort()
        
        print(f"Found {len(sm_files)} .sm files in {args.input}")
        
        for sm_file in sm_files:
            filepath = os.path.join(args.input, sm_file)
            result = analyzer.analyze_file(filepath)
            results.append(result)
            
            if args.verbose:
                if result['success']:
                    print(f"{result['file']}: CP = {result['critical_path_length']}")
                else:
                    print(f"{result['file']}: ERROR - {result['error']}")
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return
    
    # Print summary
    successful_results = [r for r in results if r['success']]
    print(f"\nSuccessfully analyzed {len(successful_results)} files")
    
    if successful_results:
        print("\nSummary:")
        print(f"Min critical path: {min(r['critical_path_length'] for r in successful_results)}")
        print(f"Max critical path: {max(r['critical_path_length'] for r in successful_results)}")
        print(f"Avg critical path: {sum(r['critical_path_length'] for r in successful_results) / len(successful_results):.1f}")
    
    # Save to CSV if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as csvfile:
            fieldnames = ['file', 'critical_path_length', 'critical_job', 'num_jobs', 'num_precedences', 'success', 'error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main() 