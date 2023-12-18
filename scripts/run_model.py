"""
Basic script to run a fitness model on a set of mutations 
"""
import argparse
from ..models.site_independent_model import SiteIndependentModel
from ..utils.scoring_utils import get_mutations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a fitness model on a set of mutations")
    parser.add_argument("--input_mutations", type=str,
                        help="file containing mutations")
    parser.add_argument("--target_seq", type=str, help="wild type sequence")
    args = parser.parse_args()

    mutations = get_mutations(args.input_mutations, args.target_seq)
    print(mutations)
