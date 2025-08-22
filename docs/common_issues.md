# Common Leopard-EM Issues and Possible Solutions

While Leopard-EM is designed to be straightforward to run across various platforms, users may still encounter common issues during installation or execution.
These challenges can arise due to differences in operating systems, dependencies, or hardware configurations.
Some of the frequently reported issues and their potential solutions are enumerated below.

## 1. Compilation Errors
Leopard-EM uses `torch.compile` to optimize performance across some steps, but this is sometimes not supported on all systems or may otherwise be unstable.
The easiest way to circumvent these issues is to disable compilation by setting the environment variable `LEOPARDEM_DISABLE_TORCH_COMPILATION` before running Leopard-EM.

```bash
export LEOPARDEM_DISABLE_TORCH_COMPILATION=1
```

!!! warning "No compilation leads to decreased performance"
    
    Disabling compilation may lead to slower execution times, especially for the `match_template` program.
    If performance is critical in your case, try some basic troubleshooting or open an issue on the GitHub page if compilation errors persist.