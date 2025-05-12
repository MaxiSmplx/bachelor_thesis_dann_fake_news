#imports
import os


#constants
DATASETS = [dataset for dataset in os.listdir('prepare_datasets') if os.path.isdir(os.path.join('prepare_datasets', dataset))]
