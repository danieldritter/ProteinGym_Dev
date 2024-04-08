from omegaconf import DictConfig, OmegaConf,open_dict
import hydra
import os 
import pandas as pd 
from proteingym.utils.scoring_utils import get_mutations
from proteingym.utils.datasets import MutationDataset
import lightning as L 
import torch 

@hydra.main(version_base=None, config_path=f"{os.path.dirname(os.path.dirname(__file__))}/configs", config_name="default_supervised_config")
def main(config: DictConfig):
    ref_df = pd.read_csv(config.reference_file)
    mut_file = config.data_folder + os.sep + ref_df["DMS_filename"][config.experiment_index]
    mutations = get_mutations(mut_file, str(ref_df["target_seq"][config.experiment_index]))
    model = hydra.utils.instantiate(config.supervised_model)
    dataset = MutationDataset.from_df(mutations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    trainer = L.Trainer()
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()