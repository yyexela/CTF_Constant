## CTF Constant

This repository contains code that predicts a constant value for provided test matrices. The supported methods are:
- Predict a constant obtained from a grid-search.
- Predict the average.
- Predict zero.

## Environment

The environment is described in `requirements.txt`.

## Running

There are three configuration files:
- `config/config1.yaml`: Do a random search.
- `config/config2.yaml`: Predict the average.
- `config/config3.yaml`: Predict zero.

To run one of the configuration files, you enter the following in the command line:  
`python src/main.py --config config/config3.yml`

## Directory structure

```text
.
├── config
│   ├── config1.yml
│   ├── config2.yml
│   └── config3.yml
├── README.md
├── requirements.txt
├── results
│   ├── avg_PDE_KS_4.npy
│   ├── avg_PDE_KS_4.yml
│   ├── random_search_PDE_KS_4.npy
│   ├── random_search_PDE_KS_4.yml
│   ├── zero_PDE_KS_4.npy
│   └── zero_PDE_KS_4.yml
├── scripts
│   └── main.py
└── src
    └── helpers.py
```

- `config`: Stores configuration files.
- `README.md`: This file.
- `requirements.txt`: Specifies environment.
- `results`: Output from runs.
  - Contains yaml showing which configuration was used, as well as the predicted matrix.
- `scripts`: Contains primary run script.
- `src`: Contains helper functions.