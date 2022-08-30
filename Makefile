PYTHON := python
TENSORBOARD := tensorboard
BLACK := black

help:
	@echo "Targets:"
	@echo "* tensorboard: launch tensorboard"
	@echo "* black: reformat by black"
	@echo ""

.PHONY: tensorboard
tensorboard:
	${TENSORBOARD} --logdir tb_logs/ --bind_all --port 6006

.PHONY: black
black:
	${PYTHON} -m ${BLACK} -l 120 -t py37 src
