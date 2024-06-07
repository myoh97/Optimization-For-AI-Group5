You can either evaluate optimizers by training from scratch, or by directly using our train history.
# Train Model from Scratch
## Setting Environment

---
## Preparing Dataset
You can get dataset from ~
---
## Run Training
As pytorch is installed, you can train re-id module by running:

> bash train_{dataset_name}.sh
---

# Utilize Train History
Download our history .pth file from ~

Put file in format of ~

---
# Visuaize Learning Curves
If you want to draw results for every optimizers, then run
> python draw.py
If you want to draw results for comparing the effect of batch size, then run
> python draw_batchsize.py
