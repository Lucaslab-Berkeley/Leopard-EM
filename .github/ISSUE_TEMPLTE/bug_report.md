---
name: 🐛 Bug Report
about: Create a report to help us reproduce and fix the bug
title: "[Bug] <your descriptive title here>"
labels: bug
assignees: ''
---

## Bug Description

Please include a clear and concise description of the bug.

---

<!-- ## Checklist before submitting

I have

- [ ] Checked the open issues -->

## Platform and system information

Please run the following commands and replace the fields with the command output

* Operating System: (linux, macOS, Windows)
* OS version (Linux only): `lsb_release -a`
* Python version: `python --version`
* Leopard-EM version: `pip show leopard-em | grep Version`
* PyTorch version: `python -c 'import torch; print(torch.__version__)'`
* Nvidia driver version: `nvidia-smi --query-gpu=driver_version --format=csv,noheader`

---

## Steps to Reproduce

Please describe your expected behavior, observations, and the steps to reproduce the bug.

### Expected behavior

What did you expect to happen?

### Observed behavior

What happened instead?
If there is a traceback, please include it here or attach it as a file.

### Steps to reproduce

Please provide the steps to reproduce the bug such as:
1. Run the command `...`
2. Run this Python script/code snippet `...`
3. Run this Jupyter notebook `...`


## Additional context

Please add any other context about the problem here.
