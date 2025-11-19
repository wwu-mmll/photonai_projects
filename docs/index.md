# PHOTONAI Projects Documentation

Welcome to the **PHOTONAI Projects** toolbox — a lightweight framework for organizing, running, and statistically comparing PHOTONAI analyses.

This package helps you:

- create and manage multiple analyses inside a structured project folder  
- run PHOTONAI Hyperpipes on stored datasets  
- perform permutation tests (locally or on SLURM clusters)  
- aggregate permutation results and compute p-values  
- compare multiple analyses statistically  
  - **Nadeau–Bengio corrected t-test**  
  - **Permutation-based tests**

Whether you're running a single experiment or dozens of feature-selection pipelines, this toolbox keeps everything organized, reproducducible, and statistically valid.

---

## Quick links

- **[Getting started](getting-started.md)**
- **[API Reference](api.md)**

---

## Installation

```bash
pip install photonai photonai-projects
```

Or, if installing from a local checkout:

```bash
pip install -e .
```

---

## What’s inside the toolbox?

### **Project Management**

Store each analysis in a clean folder structure:

```
project/
├── analysis_1/
│   ├── data/
│   ├── hyperpipe_constructor.py
│   └── hyperpipe_meta.json
├── analysis_2/
└── ...
```

### **Execution**

Run any analysis:

```python
project.run("analysis_name")
```

### **Permutation Tests**

Generate permutation-based null distributions:

```python
project.run_permutation_test("analysis_name", n_perms=1000)
```

### **Statistical Comparison**

Compare analyses using:

- `method="nadeau-bengio"`
- `method="permutation"`

```python
project.compare_analyses("A", "B", method="permutation")
```

### **SLURM Support**

Automatically prepare SLURM array jobs for large permutation workloads.

---

## Additional pages coming soon

- Usage Guide  
- Writing hyperpipe constructors  
- Best practices  
- Reproducibility checklist
