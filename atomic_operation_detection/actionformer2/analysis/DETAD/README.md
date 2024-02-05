# Diagnosing Error in Temporal Action Detectors (DETAD)
This codebase is borrowed from [DETAD](https://github.com/HumamAlwassel/DETAD). 

## Installation 
Please refer to [the original codebase](https://github.com/HumamAlwassel/DETAD?tab=readme-ov-file#how-to-run-it).

## Run
First of all, convert a pickle file outputted from ActionFormer.  
  You can put "verb" or "manipulatd" or "affected" or "atomic_operation" in `[type_name]`.

  ```sh
  cd ../..
  python libs/utils/convert_pkl2json.py 
  --pkl [path/to/pkl/file]
  --label_to_int data/annotations/[type_name]_to_int.json
  --output [path/to/output/directory]
  ```

<details>
<summary>False Positive Analaysis</summary>
  Run the command below by indicating a coverted json file path with `--prediction_filename`.

  ```sh
  python src/false_positive_analysis.py
  --ground_truth_filename ../../data/annotations/annotation_all.json
  --subset test
  --prediction_filename [/path/to/json/file]
  --output_folder [/path/to/output/directory]
  --type [type_name]
  --tiou_thresholds 0.5
  ```

</details>

<details>
<summary>False Negative Analaysis</summary>
  Run the command below by indicating a coverted json file path with `--prediction_filename`.

  ```sh
  python src/false_negative_analysis.py
  --ground_truth_filename ../../data/annotations/annotation_all.json
  --subset test
  --prediction_filename [/path/to/json/file]
  --output_folder [/path/to/output/directory]
  --type [type_name]
  --tiou_thresholds 0.5
  ```

</details>


