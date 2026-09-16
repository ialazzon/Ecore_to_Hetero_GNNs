# Automating the Conversion of Ecore Models to Heterogeneous Graph Neural Networks for Model-Driven Engineering

This repository contains the source code, datasets, and experimental scripts associated with the paper:

> **“Automating the Conversion of Ecore Models to Heterogeneous Graph Neural Networks for Model-Driven Engineering”**
> **Authors:** Alaa Alhusban and Issam Al-Azzoni

The repository provides the implementation of the complete pipeline for converting Ecore model instances into heterogeneous graph representations, training a Graph Neural Network (GNN) classifier, and comparing its performance with a Random Forest (RF) classifier operating on flattened sensor data.

## Requirements

The file `requirements.txt` contains the Python libraries required to execute the scripts.

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Running the Complete Pipeline

The following steps should be executed in order.

### 1. Convert Ecore Model Instances to Heterogeneous Graphs

Run:

```bash
python convert_batch_emf_model_to_heterodata.py
```

This command converts the snapshot models stored in the `xmi_new` folder into PyTorch `.pt` graph representations.

The generated graphs are stored in the `pt_graphs` folder.

### 2. Generate the Label File

Run:

```bash
python make_label_csv.py
```

This command generates the label file:

```text
label_of.csv
```

The file contains the class labels associated with the generated graphs.

### 3. Train and Evaluate the GNN Classifier

Run:

```bash
python train_graph_clf_multiclass_saved_splits.py
```

This command trains the GNN classifier using predefined training, validation, and test graph splits.

The graph split information is stored in:

```text
exp_splits.txt
```

The evaluation results are displayed at the end of the execution.

### 4. Generate the Flattened Sensor Dataset

Run:

```bash
python flatten_sensors_to_csv.py xmi_new --ecore iotbuilding.ecore --out flattened_data.csv
```

This command generates a flattened tabular representation of the sensor readings from the XMI model instances.

The resulting file is:

```text
flattened_data.csv
```

Only sensor readings are included in this flattened representation.

### 5. Train and Evaluate the Random Forest Classifier

Run:

```bash
python rf_pipeline_fixed_splits.py
```

This command trains a Random Forest (RF) classifier using the same data splits used for the GNN experiments.

The evaluation results are displayed at the end of the execution.

## Pipeline Overview

```text
XMI Model Instances (xmi_new)
          |
          v
convert_batch_emf_model_to_heterodata.py
          |
          v
Heterogeneous Graphs (pt_graphs)
          |
          +-------------------------------+
          |                               |
          v                               v
make_label_csv.py               flatten_sensors_to_csv.py
          |                               |
          v                               v
     label_of.csv                  flattened_data.csv
          |                               |
          v                               v
GNN Classification               Random Forest Classification
          |                               |
          v                               v
train_graph_clf_multiclass_     rf_pipeline_fixed_splits.py
saved_splits.py
          |                               |
          +---------------+---------------+
                          |
                          v
                  Evaluation Results
```

## Main Repository Files

```text
Ecore_to_Hetero_GNNs/
|
|-- xmi_new/
|   XMI snapshot models used in the experiments
|
|-- pt_graphs/
|   Generated heterogeneous graph representations
|
|-- iotbuilding.ecore
|   Ecore metamodel used in the case study
|
|-- convert_batch_emf_model_to_heterodata.py
|   Converts XMI model instances into heterogeneous graphs
|
|-- make_label_csv.py
|   Generates graph labels
|
|-- train_graph_clf_multiclass_saved_splits.py
|   Trains and evaluates the GNN classifier
|
|-- flatten_sensors_to_csv.py
|   Generates the flattened sensor dataset
|
|-- rf_pipeline_fixed_splits.py
|   Trains and evaluates the Random Forest baseline
|
|-- exp_splits.txt
|   Stores the predefined experimental data splits
|
|-- requirements.txt
|   Python dependencies
|
`-- README.md
```

## Reproducibility

The experiments use predefined data splits stored in `exp_splits.txt`.

The same splits are used for both the GNN and Random Forest experiments to ensure a consistent comparison between the graph-based and tabular classification approaches.

## Citation

If you use this repository or its implementation in your research, please cite the associated paper:

```text
Alaa Alhusban and Issam Al-Azzoni,
"Automating the Conversion of Ecore Models to Heterogeneous Graph Neural
Networks for Model-Driven Engineering."
```

Full publication information will be added once the paper is published.

## License

This repository is distributed under the terms of the license provided in the `LICENSE` file.
