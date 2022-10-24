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

## Preprocess
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

In the following link you can find the full Controlled Text Reduction dataset: [link](https://huggingface.co/datasets/biu-nlp/Controlled-Text-Reduction-dataset)