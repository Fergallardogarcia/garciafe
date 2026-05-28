.PHONY: deploy_all tsne architecture

# Default variables (can be overridden via command line)

configs_dir := c-GAN_code/configs
dataset := CIFAR10# [MNIST, CIFAR10]

run-%:
	$(MAKE) run EXP_TYPE=$*

run:
	@FILES=$$(ls -1 $(configs_dir)/$(dataset)/*$$EXP_TYPE*.yaml 2>/dev/null); \
	N=$$(echo "$$FILES" | wc -w); \
	if [ $$N -eq 0 ]; then echo "No configs found"; exit 1; fi; \
	ARRAY="0-$$(($$N-1))"; \
	JOB_NAME=$$EXP_TYPE; \
	sed \
	    -e "s/#SBATCH -J SAVE/#SBATCH -J $$JOB_NAME/" \
	    -e "s/EXP_TYPE=\"SAVE\"/EXP_TYPE=\"$$EXP_TYPE\"/" \
	    -e "s/#SBATCH --array=.*/#SBATCH --array=$$ARRAY/" \
	    -e "s|configs/\*|configs/$(dataset)/*|" \
	    deploy_all.slurm | sbatch

tsne:
	sbatch TSNE.slurm

plot_container := /cephyr/users/garciafe/containers/fl_env_v5.sif
plot_script := scripts/plot_gen_corr_pert_ratio_log.py
plot_run_results ?= /cephyr/users/garciafe/temp/CIFAR10/run_results
plot_output ?= $(plot_run_results)/filter_gen_corr_pert_ratio_trends_minibatch_axes_no_discrim_log.png

architecture:
	apptainer exec $(plot_container) python3 $(plot_script) \
		--run-results-path $(plot_run_results) \
		--output $(plot_output)

queue:
	squeue -u $$USER

cancel:
	scancel -u $$USER