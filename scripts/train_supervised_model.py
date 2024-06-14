from omegaconf import DictConfig, OmegaConf,open_dict
import hydra
import os 
import pandas as pd 
from proteingym.utils.scoring_utils import get_mutations
from proteingym.utils.datasets import MutationDataset
import lightning as L 
import torch 
import numpy as np 
import random 

@hydra.main(version_base=None, config_path=f"{os.path.dirname(os.path.dirname(__file__))}/configs", config_name="default_supervised_config")
def main(config: DictConfig):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    ref_df = pd.read_csv(config.reference_file)
    mut_file = config.data_folder + os.sep + ref_df["DMS_filename"][config.experiment_index]
    mutations = get_mutations(mut_file, str(ref_df["target_seq"][config.experiment_index]))
    if "alignment" in config:
        if "MSA_filename" in ref_df.columns:
            assert "alignment_folder" in config, "Must provide alignment_folder in config if MSA_filename is provided"
            config.alignment["alignment_file"] = config.alignment_folder + os.sep + ref_df["MSA_filename"][config.experiment_index]
        if "weight_file_name" in ref_df.columns:
            assert "weights_folder" in config, "Must provide weights_folder in config if weight_file_name is provided"
            with open_dict(config.alignment):
                config.alignment["weights_file"] = config.weights_folder + os.sep + ref_df["weight_file_name"][config.experiment_index]
        # Adding alignment_kwargs field to config for base model so that hydra's recursive instantiation will include alignment parameters
        with open_dict(config.supervised_model.model):
            config.supervised_model.model["alignment_kwargs"] = config.alignment
    model = hydra.utils.instantiate(config.supervised_model)
    dataset = MutationDataset.from_df(mutations)
    train_dataset, validation_dataset, test_dataset = dataset.train_val_test_split(split_type=config["mutation_split_type"])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=config["train_shuffle"])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config["eval_batch_size"], shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config["eval_batch_size"], shuffle=False)
    logger = L.pytorch.loggers.CSVLogger(save_dir=output_dir, name="")
    trainer = L.Trainer(**config.trainer, logger=logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == "__main__":
    main()