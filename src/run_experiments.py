from src.run import main as seq_to_seq_main
from src.simple_concatenation_baseline import main as simple_concatenation_main

from src.utils import prepare_config_for_hf


if __name__ == "__main__":

    config = prepare_config_for_hf()

    experiment_type = config['experiment_type']

    if experiment_type == "seq2seq":
        seq_to_seq_main()
    elif experiment_type == "simple_concatenation":
        simple_concatenation_main(config, summaries_to_test_key="simple_concatenation")
    elif experiment_type == "gold_summaries":
        simple_concatenation_main(config, summaries_to_test_key="gold_summaries")