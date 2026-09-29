"""
Shared physics core for the alpha paper. Each physical quantity is computed
in exactly one place here; figure and diagnostic scripts should only read
caches produced by pipeline/compute_all.py, never reimplement this math.
"""
