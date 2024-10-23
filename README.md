# ecs171

This repository contains the necessary materials for the ecs171 project. Below are instructions on setting up the Conda environment and installing the required dependencies.

## Prerequisites

Make sure you have Anaconda installed.

## Setting Up the Repo and Data

1. Clone the repository:

   ```git clone https://github.com/your-username/ecs171.git```

2. Download the dataset from https://drive.google.com/drive/folders/1RTILMf8xOB4rua_RgdXMv5xBvs40wfax?usp=drive_link

3. Add the "raw" folder dataset to the home directory of the repo. Make sure to keep the name of the folder as "raw".

## Setting Up the Conda Environment

1. Create a new Conda environment named `ecs171`:

   ```conda create --name ecs171 python=3.9```

2. Activate the environment:

   ```conda activate ecs171```

3. Install the required packages from the `requirements.txt` file:

   ```pip install -r requirements.txt```

## Deactivating the Conda Environment

Once you're done working in the environment, you can deactivate it with:

   ```conda deactivate```
