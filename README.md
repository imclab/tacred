# TACRED

Stanford TAC-KBP relation extraction dataset.


## Requirements

### Dataset

To use, download the TACRED dataset and unzip its contents in the root directory (thereby creating a `TACRED` folder).

### Lua dependencies

```text
torch
lualogging
torchlib
rnn
```


## Usage

The experiments are organized in a series of binaries and lua modules. Each binary has a commandline interface,
which can be described via `./bin/binary.lua -h`.

### Training from scratch

Preprocess the dataset:

```bash
./bin/serialize_dataset.lua  # insert span demarcations to create input sequences and add typecheck information
./bin/convert_dataset.lua    # convert sequences to numerical indices
```

Training:

```bash
./bin/train.lua
```

Alternatively, you can use the optional tuning script to perform random hyperparameter search. This requires
a remote postgres instance to store your data in and the lua package `hypero`.

Training will optimize with respect to the training split, perform early stopping on the development split,
and finally test on the test script.


### Post-training evaluation

You can evaluate a binarized model (which training produces automatically) via `bin/test.lua`.


### Stanford KBP

You can produce predictions for the internal Stanford KBP pipeline via

```bash
bin/query.py <args> | bin/pred.lua > extractions.tsv
```

This output file can then be loaded into the internal system as a KB table. Given the size of the internal corpus,
you can also shard the queries across multiple nodes and predict in parallel. The outputs for each shard can then
be concatenated and loaded into the internal system.
