import os
import pandas as pd


class PredictionsAnalyzer:
    def __init__(self, tokenizer, training_args) -> None:
        self.tokenizer = tokenizer
        self.training_args = training_args

    def write_predictions_to_file(self, predictions, dataset):
        decoded_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=False, clean_up_tokenization_spaces=True,
        )
        input_seqs = [self.tokenizer.decode(dataset[i]['input_ids'])
                        for i in range(len(dataset))]
        # Length can be useful to see if the model actually saw everything
        input_tokenizer_lengths = [len(dataset[i]['input_ids']) for i in range(len(dataset))]

        decoded_predictions = [pred.strip() for pred in decoded_predictions]

        def remove_pad_tokens(prediction_tokens):
            """
            We want to calculate the num of tokens without the padding
            """

            return [token for token in prediction_tokens if token != self.tokenizer.pad_token_id]

        predictions_tokenizer_lengths = [len(remove_pad_tokens(predictions[i])) for i in range(len(predictions))]


        output_prediction_file = os.path.join(
            self.training_args.output_dir, "generated_predictions.csv")

        gold = None
        if 'labels' in dataset[0]:
            gold = [self.tokenizer.decode(dataset[i]['labels'])
                                for i in range(len(dataset))]
            # Length can be useful to see if the model actually saw everything
            gold_tokenizer_lengths = [len(dataset[i]['labels']) for i in range(len(dataset))]

        # save all to dataframe
        objects = {"input": input_seqs, "input_tokenizer_length": input_tokenizer_lengths, "predicted": decoded_predictions, "prediction_tokenizer_length": predictions_tokenizer_lengths}
        if gold is not None:
            objects["gold"] = gold
            objects["gold_tokenizer_length"] = gold_tokenizer_lengths
        df = pd.DataFrame(objects)
        df.to_csv(output_prediction_file, index=False)
