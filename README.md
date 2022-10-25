# Controlled_Text_Reduction

## Getting Started 
First, clone this repository:
```
git lfs install
git clone --recursive https://github.com/lovodkin93/Controlled_Text_Reduction.git
cd Controlled_Text_Reduction
```
then run:
```
python3 -m pip install --user virtualenv
```
if you haven't installed yet virtualenv, and then run:
```
python3 -m venv venvs/controlled_text_reduction_env
source venvs/controlled_text_reduction_env/bin/activate
python3 -m pip install -r requirements.txt
```

## Preprocess Data
The next step is to preprocess the data. To do that, run:
```
python -m src.run_experiments configs/preprocess/<CONFIG_PREPROCESS_FILE>
```
You can find examples of `<CONFIG_PREPROCESS_FILE>` under `configs/preprocess/`
Some note about the `<CONFIG_PREPROCESS_FILE>`:
1. The `train_file_highlight_rows` parameter should be a csv file with the following columns: `topic`, `summaryFile`, `documentFile`, `docSpanOffsets`, where `topic` should be identical to `summaryFile`, and the `summaryFile`, `documentFile` columns should include the names of the extracted documents and summaries (see the next point).
2. The `doc_data_dir` parameter should point to the datapath, which should contain the following structure where a summary and its related document directory share the same name:
```
      - <DATA_PATH>
        - summaries
          - A.txt
          - B.txt
          - ...
        - A
          - doc_A1
          - doc_A2
          - ...
        - B
          - doc_B1
          - doc_B2
          - ...
```
3. The `output_file_path` parameter should save the output file in the `data/` folder.

In the following link you can find the full Controlled Text Reduction dataset: [link to dataset](https://huggingface.co/datasets/biu-nlp/Controlled-Text-Reduction-dataset)

## Train Models
Once the datasets have been preprocessed and saved in the `data/` folder, you can proceed to training the models.
To do that, run:
```
python -m src.run_experiments configs/train/<CONFIG_TRAIN_FILE>
```
You can find examples of `<CONFIG_TRAIN_FILE>` under `configs/train/`. 
You have different options:
1. To train a model which receives as input text+highlights, follow `train_led__4096__global_text_and_highlights.json`.
2. To train a model which receives as input only text (without highlights), follow `train_led__4096__no_highlights.json`.
3. To train a model which receives as input only highlights (their concatenation), follow `train_led__4096__global_only_highlights.json`.
4. To further finetune an already finetuned model (e.g., dirst finetuning on CNN-DM and DUC training set and then further finetuning on DUC alone), follow `further_finetune_led_4096_global_on_highlights_pretrained_CNNDM_full_and_duc.json`.

## Eval and Test Models
To test or evaluate a trained model, run:
```
python -m src.run_experiments configs/eval/<CONFIG_TEST_FILE>
```
You can find examples of `<CONFIG_TEST_FILE>` under `configs/eval/`. 
You have different options:
1. To test a simple concatenation, follow `test_simple_concatenation.json` (or `eval_simple_concatenation.json` for the devset).
2. To test a model which receives as input text+highlights, follow `test_led_text_and_highlights.json` (or `eval_led_text_and_highlights.json` for the devset).
3. To test a model which receives as input only text (without highlights), follow `test_led_no_highlights.json` (or `eval_led_no_highlights.json` for the devset).
4. To test a model which receives as input only highlights, follow `test_led_only_highlights.json` (or `eval_led_only_highlights.json` for the devset).
5. To run a mixed experiment, where highlights are paired with a different summary of the same document, follow `test_led_mixed.json`.

Citation
========
If you find this repository useful in your research, please cite the following paper:
```
@misc{https://doi.org/10.48550/arxiv.2210.13449,
  doi = {10.48550/ARXIV.2210.13449},
  url = {https://arxiv.org/abs/2210.13449},
  author = {Slobodkin, Aviv and Roit, Paul and Hirsch, Eran and Ernst, Ori and Dagan, Ido},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Controlled Text Reduction},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Zero v1.0 Universal}
}

```
