from omegaconf import DictConfig, OmegaConf,open_dict
import hydra
import os 
import pandas as pd 
from proteingym.utils.scoring_utils import get_mutations
from proteingym.utils.datasets import MutationDataset
import lightning as L 
import torch 
import numpy as np 

@hydra.main(version_base=None, config_path=f"{os.path.dirname(os.path.dirname(__file__))}/configs", config_name="default_supervised_config")
def main(config: DictConfig):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    ref_df = pd.read_csv(config.reference_file)
    mut_file = config.data_folder + os.sep + ref_df["DMS_filename"][config.experiment_index]
    mutations = get_mutations(mut_file, str(ref_df["target_seq"][config.experiment_index]))
    model = hydra.utils.instantiate(config.supervised_setup.supervised_model)
    dataset = MutationDataset.from_df(mutations)
    train_dataset, validation_dataset, test_dataset = dataset.train_val_test_split(split_type=config.supervised_setup["mutation_split_type"])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.supervised_setup["train_batch_size"], shuffle=config.supervised_setup["train_shuffle"])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.supervised_setup["eval_batch_size"], shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config.supervised_setup["eval_batch_size"], shuffle=False)
    logger = L.pytorch.loggers.CSVLogger(save_dir=output_dir, name="")
    trainer = hydra.utils.instantiate(config.supervised_setup.trainer, logger=logger)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == "__main__":
    main()