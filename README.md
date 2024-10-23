# Waffle
WAFFLE: Webpage Multi-Modal Model for Automated Front-End Development

## Dependency
- peft               0.11.1
- transformers       4.41.1
- pytorch       2.3.0
- selenium
- Python 3.10.14
- deepspeed          0.14.1
- datasets 2.19.1
- beautifulsoup4     4.12.3
- accelerate         0.30.1

## Structure
- `moondream_finetune` contains the dataset class file, model class files, and training file for moondream2.
    - `moondream2` contains model class files. Specifically `modeling_web.py` is our model class file and `generation_utils.py` is modified for inference with attention.
    - `finetune.py` is the training file
    - `eval_moondream.py` is the inference file
    - `dataset.py` is the dataset class file

- `vlm_websight` contains the dataset class file, model class files, and training file for vlm_websight.
    - `VLM_WebSight` contains model class files. Specifically `modeling_web.py` is our model class file and `generation_utils.py` is modified for inference with attention.
    - `finetune.py` is the training file
    - `eval_websight.py` is the inference file
    - `dataset.py` is the dataset class file
- html.tar contains all the html source code in our mutation dataset. To unzip, use: `tar -I "zstd" -xf html.tar.zst`

- Waffle-Test is one of our test dataset
- Design2Code-Test is the other test dataset

## Usage
To run our code, you can render the HTML code to images, and then load and tokenize them according the dataset.py files.

To train, use the finetune.py files in each directory, replace the paths accordingly. E.g., pre-trained ckpt path, save_path, dataset path.

