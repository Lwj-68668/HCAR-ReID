# Dataset Preparation

This project uses existing public benchmarks:

- MARS
- iLIDS-VID
- PRID2011

Raw datasets are not redistributed in this repository. Follow the official dataset licenses and terms.

## Required Local Root

All dataset readers assume the root folder:

`./Data`

## 1) MARS Layout

The current loader expects this exact structure:

```text
Data/
  MARS/
    info/
      train_name.txt
      test_name.txt
      tracks_train_info.mat
      tracks_test_info.mat
      query_IDX.mat
    bbox_train/
      bbox_train/
        0001/
          0001C1T0001F001.jpg
          ...
    bbox_test/
      bbox_test/
        0001/
          0001C1T0001F001.jpg
          ...
```

Important: the path includes a double folder level (`bbox_train/bbox_train` and `bbox_test/bbox_test`) because that is what the current parser builds.

## 2) iLIDS-VID Layout

```text
Data/
  i-LIDS-VID/
    sequences/
      cam1/
        person_XXX/
          *.png
      cam2/
        person_XXX/
          *.png
    train-test people splits/
      train_test_splits_ilidsvid.mat
    splits.json
```

Notes:

- `splits.json` can be generated automatically from `train_test_splits_ilidsvid.mat`.
- In the current code, auto-download is triggered only when `./Data` does not exist. If `./Data` already exists, place iLIDS-VID manually.

## 3) PRID2011 Layout

```text
Data/
  prid_2011/
    multi_shot/
      cam_a/
        person_XXX/
          *.png
      cam_b/
        person_XXX/
          *.png
    splits_prid2011.json
```

`splits_prid2011.json` is required by the parser.

## Quick Sanity Checks

Run one of the commands below after placing data:

```bash
python train_hierarchical.py --Dataset_name Mars --epochs 1 --batch_size 2 --seq_len 4 --save_dir tmp_check --seed 42
python train_hierarchical.py --Dataset_name iLIDSVID --epochs 1 --batch_size 2 --seq_len 4 --save_dir tmp_check --seed 42
python train_hierarchical.py --Dataset_name PRID --epochs 1 --batch_size 2 --seq_len 4 --save_dir tmp_check --seed 42
```

If paths are correct, dataset statistics will print before training starts.
