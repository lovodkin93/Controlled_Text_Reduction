import os
import sys
from datasets import load_metric
import pandas as pd
import numpy as np    
import json


class PredictionsAnalyzer:
    def __init__(self, tokenizer, output_dir: str) -> None:
        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def write_predictions_to_file(self, predictions, dataset, is_tokenized=True):

        def remove_pad_tokens(prediction_tokens):
            """
            We want to calculate the num of tokens without the padding
            """

            return [token for token in prediction_tokens if token != self.tokenizer.pad_token_id]

        # Non-tokenized can be outputs not from a model, such as naive concatenation
        if not is_tokenized:
            decoded_predictions = predictions
            input_seqs = dataset
            input_tokenizer_lengths = None
            predictions_tokenizer_lengths = None
        else:
            decoded_predictions = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            decoded_predictions = [pred.strip() for pred in decoded_predictions]

            input_seqs = [self.tokenizer.decode(dataset[i]['input_ids'])
                            for i in range(len(dataset))]

            # Length can be useful to see if the model actually saw everything
            predictions_tokenizer_lengths = [len(remove_pad_tokens(predictions[i])) for i in range(len(predictions))]
            input_tokenizer_lengths = [len(dataset[i]['input_ids']) for i in range(len(dataset))]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        output_prediction_file = os.path.join(
            self.output_dir, "generated_predictions.csv")

        gold = None
        if 'labels' in dataset[0]:
            gold = [self.tokenizer.decode(dataset[i]['labels'], skip_special_tokens=True)
                                for i in range(len(dataset))]
            # Length can be useful to see if the model actually saw everything
            gold_tokenizer_lengths = [len(dataset[i]['labels']) for i in range(len(dataset))]

        # save all to dataframe
        objects = {"input": input_seqs, "input_tokenizer_length": input_tokenizer_lengths, "predicted": decoded_predictions, "prediction_tokenizer_length": predictions_tokenizer_lengths}
        if gold is not None:
            objects["gold"] = gold
            objects["gold_tokenizer_length"] = gold_tokenizer_lengths

            self.calculate_rouge_between_gold_n_prediction(objects, decoded_predictions, gold)

        if not is_tokenized:
            clean_input_seqs = dataset
        else:
            clean_input_seqs = [self.tokenizer.decode(dataset[i]['input_ids'], skip_special_tokens=True)
                            for i in range(len(dataset))]
        
        self.calculate_summc_between_input_n_summaries(objects, clean_input_seqs, decoded_predictions)

        df = pd.DataFrame(objects)
        df.to_csv(output_prediction_file, index=False)

    def calculate_rouge_between_gold_n_prediction(self, objects, decoded_predictions, gold):
        # Add rouge per prediction
        metric = load_metric("rouge")
        result_per_pred = metric.compute(predictions=decoded_predictions, references=gold, use_stemmer=True, use_aggregator=False)
        objects['rouge1'] = [x.fmeasure for x in result_per_pred['rouge1']]
        objects['rouge2'] = [x.fmeasure for x in result_per_pred['rouge2']]

    def calculate_summc_between_input_n_summaries(self, objects, inputs, summaries):
        sys.path.append('summac')  # Will fail if you didn't load the submodule (https://git-scm.com/book/en/v2/Git-Tools-Submodules)
        from summac.model_summac import SummaCZS
        model = SummaCZS(granularity="sentence", model_name="vitc", use_con=False)

        result = model.score(inputs, summaries)

        per_example_per_sentence_highest_source_score = []
        for example_idx, image in enumerate(result['images']):
            split_input = model.imager.split_text(inputs[example_idx])
            split_summary = model.imager.split_text(summaries[example_idx])
            per_sentence_highest_source_score = []
            # Image shape: 3 x num_source_sents x num_summary_sents
            for summary_sentence_idx in range(0, image.shape[2]):
                summary_sentence = split_summary[summary_sentence_idx]
                max_ent_score, max_ent_idx, max_con_score, max_con_idx, final_score = self._summc_from_image_to_scores(model, image[:,:,summary_sentence_idx])

                per_sentence_highest_source_score.append({
                    "hypothesis": summary_sentence,
                    "score": final_score,
                    "max_ent_score": max_ent_score,
                    "max_ent_premise": split_input[max_ent_idx],
                    "max_con_score": max_con_score,
                    "max_con_premise": split_input[max_con_idx],
                })

            per_example_per_sentence_highest_source_score.append(json.dumps(per_sentence_highest_source_score))

        objects['summac_per_example_per_sentence_highest_source_score'] = per_example_per_sentence_highest_source_score
        objects['summac_scores'] = result['scores']

    def _summc_from_image_to_scores(self, model, image):
        """
        Copy pasted from summac with modifications to run over a single summary sentence
        """

        ent_scores = image[0]  # Shape: num_input_sentences
        con_scores = image[1]  # Shape: num_input_sentences

        max_ent_idx = np.argmax(ent_scores, axis=0)
        max_ent_score = np.max(ent_scores, axis=0)
        max_con_idx = np.argmax(con_scores, axis=0)
        max_con_score = np.max(con_scores, axis=0)


        if model.use_ent and model.use_con:
            final_score = max_ent_score - max_con_score
        elif model.use_ent:
            final_score = max_ent_score
        elif model.use_con:
            final_score = 1.0 - max_con_score

        return max_ent_score, max_ent_idx, max_con_score, max_con_idx, final_score
