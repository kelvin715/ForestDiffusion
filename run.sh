cd /proj-vertical-llms-pvc/users/zhihan/tabular_gen/ForestDiffusion
out_path="/proj-vertical-llms-pvc/users/zhihan/tabular_gen/ForestDiffusion/results.txt"
# out_path="/proj-vertical-llms-pvc/users/zhihan/tabular_gen/ForestDiffusion/results_mixed_flow.txt"
myargs=" --methods forest_diffusion --diffusion_type flow --out_path ${out_path} --datasets iris"
python script_generation.py ${myargs}