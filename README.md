# Diva

In-progress work for the Diva (**D**iversity **i**n **v**erific**a**tion) Coq proof script synthesis tool.

The Diva technique and its evaluation are described in Diversity-Driven Automated Verification by Emily First and Yuriy Brun. Published in the ACM/IEEE International Conference on Software Engineering (ICSE) 2022.  https://doi.org/10.1145/3510003.3510138

We have made available a replication package for the results in the paper. It is a VirtualBox VM: [here](https://doi.org/10.5281/zenodo.5903318).

The following are the directions for installation and use of Diva:

## 1. Installation

Diva operates within the [CoqGym](https://github.com/princeton-vl/CoqGym) learning environment and so modifies their code. 
The following are the dependencies and directions to install CoqGym:

### Dependencies
* [OPAM](https://opam.ocaml.org/)
* [Anaconda Python 3](https://www.anaconda.com/distribution/)
* [LMDB](https://symas.com/lmdb/)
* [Ruby](https://www.ruby-lang.org/en/)


### Building Coq, SerAPI, CoqHammer, and the CoqGym Coq Projects

1. Create an OPAM switch for OCaml 4.07.1+flambda: `opam switch create 4.07.1+flambda && eval $(opam env)`
2. Upgrade the installed OPAM packages (optional): `opam upgrade && eval $(opam env)`
3. Clone the repository: `git clone https://github.com/princeton-vl/CoqGym`
4. Install Coq, SerAPI and CoqHammer: `cd CoqGym && source install.sh`
5. Build the Coq projects (can take a while): `cd coq_projects && make && cd ..`
6. Create and activate the conda environment: `conda env create -f coq_gym.yml && conda activate coq_gym`

## 2. Extracting proofs from Coq projects

For any Coq project that compiles in Coq 8.9.1 that you want to use (and may not be in the CoqGym dataset), the following are the steps to extract the proofs from code:

1. Copy the project into the  `coq_projects` directory. 
2. For each `*.meta` file in the project, run `python check_proofs.py --file /path/to/*.meta`   
This generates a `*.json` file in `./data/` corresponding to each `*.meta` file. The `proofs` field of the JSON object is a list containing the proof names.
3. For each `*.meta` file and each proof, run:  
`python extract_proof.py --file /path/to/*.meta --proof $PROOF_NAME`  
`python extract_synthetic_proofs.py --file /path/to/*.meta --proof $PROOF_NAME`
4. Finally, run `python postprocess.py`

## 3. Using the CoqGym benchmark dataset

### Download the CoqGym dataset

1. Download the CoqGym dataset (you do not need to extract proofs from Coq projects)
[here](https://drive.google.com/drive/folders/149m_17VkYYkl0kdSB4AI8zodCuTmPaA6?usp=sharing)
2. Unzip the data and set the paths: `python unzip_data.py`

### Training Examples (proof steps)

1. Proofs steps used in the paper are found in `processed.tar.gz`, which can be downloaded from the replication package link provided above. This should be copied into `Diva/`
2. To extract new proofs, run `python extract_proof_steps.py`.
3. To generate new proof steps that have `prev_tokens` field, run `python process_proof_steps.py`. This will generate `processed/proof_steps`.

## 4. Training Diva

To train, for example, the Tok model on all the proof steps, run 
`python main.py --no_validation --exp_id tok` 

Model checkpoints will be saved in `Diva/runs/tok/checkpoints/`. See `options.py` for command line options.

## 5. Evaluation

Now, you can evaluate a model you trained on the test set. For example, the Tok model that you trained can be run with `python evaluate.py ours tok-results --path /path/to/tok_model/*.pth`.
If you used additional options in training, specify those same options for evaluating.

This command runs the model for the entire specified test set. You can specify file names, proof names, project/file indices, etc. 
