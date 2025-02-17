# Disentangling Detail from Structure in Brain MRIs 

This repository contains the code for our project in the Advanced Medical Machine Learning Seminar at HPI.
We wanted to find out if providing segmentation masks as additional input to Variational Autoencoders can help to capture more relevant details for dementia prediction and the prediction of multiple health-related traits from the UK Biobank, such as blood pressure, body mass index (BMI), and cholesterol levels.

To learn more about the project and our findings, please read the report in `report/report.pdf`.

## Prerequisites

In order to run this code you need access to a subset of the ADNI dataset and UKBB dataset which includes segmentation masks of the brain scans.
The Fachgebiet Lippert of HPI has this dataset available on the HPC of the Dhclab. Checkout `src/adni.csv` and `src/ukbb.csv` to see where the dataset is stored.

### Setup and installation

You find all the required packages in `requirements.txt`. We recommend using a conda virtual environment.

## Running the code

With this code you can...

- Train a VAE with a lot of different configurations
- Running hyperparameter sweeps from wandb
- Train a dementia classifier which uses latent representations from a VAE as input
- Train a model pipeline which trains a VAE with a lot of different configurations and performs downstream task on both ADNI data and UKBB data

### Training VAEs

Run `python src/main.py --help` to get an overview of the configuration options for the VAE.
The most important one is `model.use_segmentation_masks` which has three different options:

- "no_mask": This will train a standard VAE.
- "in_encoder": Here, segmentation masks are added as additional layers to the input of the encoder.
- "separate_encoder": Here, a separate encoder takes care of the segmentation masks. The latent variables of both encoders are then concatenated to serve as input to the decoder.

You can then start a job on the HPC by using our convenience script `train_vae.sh`.
Your command could look like this:

`sbatch scripts/train_vae.sh --model.use_segmentation_masks="in_encoder" --trainer.max_epochs=50`

### Running hyperparameter sweeps

With `run_sweep.sh` you can easily start several runs for your wandb hyperparameter sweep.
Simply create a sweep configuration and use the sweep id in the command below.

`sbatch --array=0-N%M scripts/run_sweep.sh sweep_id`

Replace N by the number of runs you want to run and M by the number of runs that should run in parallel.

### Training the dementia classifier

If you are done with training a VAE you can evaluate it with the downstream task of dementia prediction.

To train the downstream task, you can use the script 'downstream_module.py'. Make sure to have a checkpoint `last.ckpt` placed in `checkpoints/RUN_ID/`.
You can start the script by running:

`sbatch scripts/downstream.sh <RUN_ID> `

To use UKBB data you can start the script like this:

`sbatch scripts/downstream.sh <RUN_ID> --is_ukbb --label_column <NAME OF COLUMN YOU WANT TO PREDICT>`

Note that the output will not be logged to wandb. Instead, you will find the output in the `slogs` directory.

### Open an interactive session

If you want to create an interactive session on the cluster, you can use:

`srun -p gpu --gpus=1 --mem 16gb -c 8 --pty bash`

### Evaluating your VAE on the test set

To evaluate your final model based on unseen testing data, you can simply run

`sbatch scripts/test_vae.sh <RUN_ID>`

As before, make sure that the checkpoint you want to test is located in `checkpoints/RUN_ID/last.ckpt`.

### Runnin the model pipeline

To run the whole model pipeline you can do it like this:

`sbatch scripts/run_both.sh --data.batch_size=32 --model.use_segmentation_masks="separate_encoder" --trainer.max_epochs=30 --seed_everything <seed_nr>`

