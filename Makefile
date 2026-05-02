.PHONY: deploy_all tsne

# Default variables (can be overridden via command line)


run-%:
	$(MAKE) run EXP_TYPE=$*

run:
	@FILES=$$(ls -1 c-GAN_code/configs/*$$EXP_TYPE*.yaml 2>/dev/null); \
	N=$$(echo "$$FILES" | wc -w); \
	if [ $$N -eq 0 ]; then echo "No configs found"; exit 1; fi; \
	ARRAY="0-$$(($$N-1))"; \
	JOB_NAME=$$EXP_TYPE; \
	sed \
	    -e "s/#SBATCH -J SAVE/#SBATCH -J $$JOB_NAME/" \
	    -e "s/EXP_TYPE=\"SAVE\"/EXP_TYPE=\"$$EXP_TYPE\"/" \
	    -e "s/#SBATCH --array=.*/#SBATCH --array=$$ARRAY/" \
	    deploy_all.slurm | sbatch

tsne:
	sbatch TSNE.slurm

queue:
	squeue -u $$USER

cancel:
	scancel -u $$USER