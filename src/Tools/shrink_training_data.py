#!/usr/bin/env python3
"""
Script to create smaller versions of training data files for faster pipeline development.
Adjust the SIZE_* variables below to control subset dimensions.
"""

import os

# ============================================================================
# CONFIGURATION - Adjust these variables to control subset sizes
# ============================================================================
SIZE_SEQS = 5000          # Number of rows from training_seqs.txt
SIZE_DBPS = 50            # Number of rows from training_DBPs.txt
SIZE_DATA_ROWS = 5000     # Number of rows from training_data.txt
SIZE_DATA_COLS = 50       # Number of columns from training_data.txt

# ============================================================================
# File paths
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Data')

input_seqs = os.path.join(DATA_DIR, "training_seqs.txt")
input_dbps = os.path.join(DATA_DIR, "training_DBPs.txt")
input_data = os.path.join(DATA_DIR, "training_data.txt")

output_seqs = os.path.join(DATA_DIR, "training_seqs_small.txt")
output_dbps = os.path.join(DATA_DIR, "training_DBPs_small.txt")
output_data = os.path.join(DATA_DIR, "training_data_small.txt")

# ============================================================================
# Helper function to shrink files
# ============================================================================
def shrink_file_rows(input_path, output_path, num_rows, description=""):
    """Read first N rows from input file and write to output file."""
    print(f"Shrinking {description or input_path}...")
    try:
        with open(input_path, 'r') as infile:
            rows = [next(infile).rstrip('\n') for _ in range(num_rows)]
        
        with open(output_path, 'w') as outfile:
            outfile.write('\n'.join(rows) + '\n')
        
        print(f"  ✓ Created {output_path} with {len(rows)} rows")
        return True
    except FileNotFoundError:
        print(f"  ✗ Error: {input_path} not found")
        return False
    except Exception as e:
        print(f"  ✗ Error processing {input_path}: {e}")
        return False

def shrink_file_rows_cols(input_path, output_path, num_rows, num_cols, description=""):
    """Read first N rows and M columns from input file (space/tab-separated)."""
    print(f"Shrinking {description or input_path}...")
    try:
        rows_processed = []
        with open(input_path, 'r') as infile:
            for i, line in enumerate(infile):
                if i >= num_rows:
                    break
                # Split by whitespace and take first N columns
                parts = line.strip().split()
                selected_cols = parts[:num_cols]
                rows_processed.append(' '.join(selected_cols))
        
        with open(output_path, 'w') as outfile:
            outfile.write('\n'.join(rows_processed) + '\n')
        
        print(f"  ✓ Created {output_path} with {len(rows_processed)} rows and {num_cols} columns")
        return True
    except FileNotFoundError:
        print(f"  ✗ Error: {input_path} not found")
        return False
    except Exception as e:
        print(f"  ✗ Error processing {input_path}: {e}")
        return False

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Training Data Shrinking Script")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Seqs rows: {SIZE_SEQS}")
    print(f"  DBPs rows: {SIZE_DBPS}")
    print(f"  Data rows: {SIZE_DATA_ROWS}, columns: {SIZE_DATA_COLS}")
    print()
    
    success = True
    
    # Shrink seqs file
    success &= shrink_file_rows(
        input_seqs, 
        output_seqs, 
        SIZE_SEQS, 
        "training_seqs.txt"
    )
    
    # Shrink DBPs file
    success &= shrink_file_rows(
        input_dbps, 
        output_dbps, 
        SIZE_DBPS, 
        "training_DBPs.txt"
    )
    
    # Shrink data file (with column selection)
    success &= shrink_file_rows_cols(
        input_data, 
        output_data, 
        SIZE_DATA_ROWS, 
        SIZE_DATA_COLS, 
        "training_data.txt"
    )
    
    print()
    if success:
        print("✓ All files created successfully!")
    else:
        print("✗ Some files failed to process")
