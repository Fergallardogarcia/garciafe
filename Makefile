.PHONY: deploy_all tsne architecture

# Default variables (can be overridden via command line)

configs_dir := c-GAN_code/configs
dataset := CIFAR10# [FMNIS, MNIST, CIFAR10]

run-%:
	$(MAKE) run EXP_TYPE=$*

run:
	@FILES=$$(ls -1 $(configs_dir)/$(dataset)/*$$EXP_TYPE*.yaml 2>/dev/null); \
	N=$$(echo "$$FILES" | wc -w); \
	if [ $$N -eq 0 ]; then echo "No configs found"; exit 1; fi; \
	ARRAY="0-$$(($$N-1))"; \
	JOB_NAME=$$EXP_TYPE; \
	case "$(dataset)" in \
	    *CIFAR*) TIME="0-24:00:00" ;; \
	    *MNIST*|*FMNIST*) TIME="0-02:00:00" ;; \
	    *) TIME="0-24:00:00" ;; \
	esac; \
	sed \
	    -e "s/#SBATCH -J SAVE/#SBATCH -J $$JOB_NAME/" \
	    -e "s/EXP_TYPE=\"SAVE\"/EXP_TYPE=\"$$EXP_TYPE\"/" \
	    -e "s/#SBATCH --array=.*/#SBATCH --array=$$ARRAY/" \
	    -e "s|#SBATCH -t .*|#SBATCH -t $$TIME|" \
	    -e "s|configs/\*|configs/$(dataset)/*|" \
	    deploy_all.slurm | sbatch

tsne:
	sbatch TSNE.slurm

plot_container := /cephyr/users/garciafe/containers/fl_env_v5.sif
plot_script := scripts/plot_gen_corr_pert_ratio_log.py
plot_run_results ?= /cephyr/users/garciafe/temp/CIFAR10/run_results
plot_output ?= $(plot_run_results)/filter_gen_corr_pert_ratio_trends_minibatch_axes_no_discrim_log.pdf
plot_quantile ?= 0.8
# Minibatch group visibility (set at most one to 1):
#   no_constrained_minibatch=1   → hide constrained_minibatch_* (show only unconstrained)
#   no_unconstrained_minibatch=1 → hide unconstrained_minibatch_* (show only constrained)
#   both 0                       → show all minibatch files (discrimination always excluded)

constrained_minibatch := 0
unconstrained_minibatch := 1
# Set to 1 to omit the attack gen loss overlay
no_attack_gen ?= 1

ifeq ($(unconstrained_minibatch),1)
_plot_exclude_flags := --exclude-substr console_constrained_minibatch
else ifeq ($(constrained_minibatch),1)
_plot_exclude_flags := --exclude-substr unconstrained_minibatch --exclude-substr minibatch_discrimination
else
_plot_exclude_flags := --exclude-substr minibatch_discrimination
endif

ifeq ($(no_attack_gen),1)
_plot_no_atk := --no-attack-gen
else
_plot_no_atk :=
endif

architecture:
	apptainer exec $(plot_container) python3 $(plot_script) \
		--run-results-path $(plot_run_results) \
		--output $(plot_output) \
		--quantile $(plot_quantile) \
		$(_plot_exclude_flags) \
		$(_plot_no_atk)

queue:
	squeue -u $$USER

cancel:
	scancel -u $$USER