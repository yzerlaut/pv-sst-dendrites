python_path=$HOME"/miniconda3/envs/allen/bin/python" # need the "allen" environment
# 1) fetch the meshes of cell and compute the skeletons with synapses
#$python_path src/compute_and_save_meshworks.py data/cells.csv
# 2) fetch the meshes of cell and compute the skeletons with synapses
$python_path src/compute_and_save_meshworks.py MC &
$python_path src/compute_and_save_meshworks.py BC &

