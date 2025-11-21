# PHOTONAI Projects Documentation

Welcome to the **PHOTONAI Projects** toolbox — a lightweight framework for organizing, running, and statistically comparing **PHOTONAI** analyses.

This package helps you:

- create and manage multiple analyses inside a structured project folder  
- run **PHOTONAI Hyperpipes** on stored datasets  
- perform permutation tests (locally or on SLURM clusters)   
- compare multiple analyses statistically (using Nadeau-Bengio or permutations)


---

## **Project Management**

### **Project Structure**
Each PHOTONAI analysis is stored in a clean folder structure:

```
project/
├── analysis_1/
│   ├── data/
│   │   ├── X.npy
│   │   └── y.npy
│   ├── permutations/
│   │   ├── 0/
│   │   ├── 1/
│   │   └── ...
│   ├── slurm_job.cmd
│   ├── hyperpipe_constructor.py
│   └── hyperpipe_meta.json
├── analysis_2/
└── ...
```

### **Create Project**

At the beginning of your project, create a new PHOTONAI project.

```python
project = PhotonaiProject(project_folder='project')
project.add(name="analysis_name", 
            X=X, 
            y=y, 
            hyperpipe_script="path/to/hyperpipe_constructor.py",
            name_hyperpipe_constructor="create_hyperpipe")
```
### **Add Analysis**

Then add PHOTONAI analyses using .add(). 

```python
project.add(name="analysis_name", 
            X=X, 
            y=y, 
            hyperpipe_script="path/to/hyperpipe_constructor.py",
            name_hyperpipe_constructor="create_hyperpipe")
```

### **Run Analysis**

Simply use the .run() method to run a specific analysis that has already been added:

```python
project.run("analysis_name")
```

## **Statistics**

### **Permutation Test**
Generate permutation-based null distributions for a specific analysis:

```python
project.run_permutation_test("analysis_name", n_perms=1000)
```

### **Compare Analyses**

Compare two analyses using:

- `method="nadeau-bengio"`
- `method="permutation"`

```python
project.compare_analyses("analysis_1", "analysis_2", method="permutation")
```
