import os
import pandas as pd


class PredictionsAnalyzer:
    def __init__(self, tokenizer, training_args) -> None:
        self.tokenizer = tokenizer
        self.training_args = training_args

    def write_predictions_to_file(self, predictions, dataset):
        predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )
        predictions = [p for p in predictions]
        input_seqs = [self.tokenizer.decode(dataset[i]['input_ids'])
                        for i in range(len(dataset))]        

        predictions = [pred.strip() for pred in predictions]
        output_prediction_file = os.path.join(
            self.training_args.output_dir, "generated_predictions.csv")

        gold = None
        if 'labels' in dataset[0]:
            gold = [self.tokenizer.decode(dataset[i]['labels'])
                                for i in range(len(dataset))]


        # save all to dataframe
        objects = {"input": input_seqs, "predicted": predictions}
        if gold is not None:
            objects["gold"] = gold
        df = pd.DataFrame(objects)
        df.to_csv(output_prediction_file, index=False)
