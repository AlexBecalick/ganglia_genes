# Selecting in situ genes

## Installation
First, clone the repository:
```
git clone https://github.com/AlexBecalick/ganglia_genes.git
```

Next, navigate to the repo directory and create a conda environment and install dependencies by running:
```
conda create --name ganglia-genes python
```

Finally, activate the new environment and install the package itself:
```
conda activate ganglia-genes
pip install -e .
```

Then run ganglia_genes.ipynb to generate your gene list. Copy the list to sharma_scanpy.ipynb for plotting
