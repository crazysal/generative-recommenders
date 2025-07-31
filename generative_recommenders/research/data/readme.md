
---

## ✅ **Final Understanding: DatasetV2**

### 🔄 Purpose

`DatasetV2` ingests a CSV with per-user interaction histories and returns PyTorch-ready batches for training/evaluation.

### 🧩 Expected Input CSV Structure (1 row per user)

```csv
user_id,sequence_item_ids,sequence_ratings,sequence_timestamps
123,"[42, 17, 91]","[5, 4, 5]","[1592231122,1592229922,1592228822]"
```

> Each of these columns must be a **Python-evaluable list (string)** — using `eval()` to parse into a list of ints.

---

## 🧠 What Each Column Means

| Column                | Meaning                                                |
| --------------------- | ------------------------------------------------------ |
| `user_id`             | Unique ID for the user                                 |
| `sequence_item_ids`   | Ordered list of item IDs interacted with (must be int) |
| `sequence_ratings`    | Ratings (1–5 usually), aligned with `item_ids`         |
| `sequence_timestamps` | UNIX epoch timestamps, aligned with other columns      |

---

## 🧪 What `DatasetV2.__getitem__()` Returns

A dictionary with the following tensors:

```python
{
  "user_id": int,
  "historical_ids": LongTensor [L],       # input tokens (excluding last)
  "historical_ratings": LongTensor [L],   # optional, same length
  "historical_timestamps": LongTensor [L],
  "history_lengths": int,                 # true sequence length before padding
  "target_ids": int,                      # last item in sequence (label)
  "target_ratings": int,
  "target_timestamps": int,
}
```

* Sequences are **reverse-chronological**, so:

  * `target = most recent`
  * `historical = all others`

* If `chronological=True`, the `historical_*` fields are reversed back to natural order before padding.

---

## 🔁 Padding and Length Logic

* Padding happens **after** slicing out the target item.
* `padding_length = max_seq_len + 1`, so if `padding_length=51`, the historical length is 50.
* Sequences are **left-padded with `0`s** to match length.

---

## ✅ `MultiFileDatasetV2` (for massive datasets)

Used in `ml-3b` and similar, where user data is sharded across multiple files like:

```
tmp/ml-3b/16x32_0.csv
tmp/ml-3b/16x32_1.csv
...
tmp/ml-3b/16x32_15.csv
```

With metadata in:

```
tmp/ml-3b/16x32_users.csv
```

You only need this class if your data is **too large to fit in memory as a single CSV**.

---

# ✅ Recommended README.md (for preparing custom datasets)

````markdown
# Custom Dataset Preparation for Generative Recommenders

This guide explains how to prepare a custom dataset compatible with `DatasetV2` used in Meta's generative recommender codebase.

---

## ✅ Required CSV Format

Each row must correspond to a single user and have the following columns:

| Column Name | Description |
|-------------|-------------|
| `user_id`   | Unique integer identifier for the user |
| `sequence_item_ids` | List of item IDs the user has interacted with, in chronological order (as a Python string list, e.g., `"[42, 17, 91]"`) |
| `sequence_ratings` | List of ratings for each item, same length and order as `sequence_item_ids` |
| `sequence_timestamps` | List of UNIX timestamps for each item interaction |

### 📌 Sample Row

```csv
user_id,sequence_item_ids,sequence_ratings,sequence_timestamps
123,"[42, 17, 91]","[5, 4, 5]","[1592231122,1592229922,1592228822]"
````

---

## ⚙️ Configuration Parameters (in code)

When calling `DatasetV2`, the following should be set:

```python
DatasetV2(
    ratings_file='path/to/your.csv',
    padding_length=51,          # 50 tokens + 1 target
    ignore_last_n=1,            # Use last item as target
    shift_id_by=1,              # Shift item IDs to start from 1 (0 = padding)
    chronological=True,         # Preserve order (not reverse truncated)
    sample_ratio=1.0            # Keep all items; use < 1.0 for subsampling
)
```

---

## 🧪 Output of Dataset

Each batch from the dataset will yield:

* `historical_ids`: input token sequence (`[item_1, ..., item_N]`)
* `target_ids`: the item to predict next
* `history_lengths`: length before padding
* `historical_ratings`, `historical_timestamps`: optional aligned metadata
* `user_id`: preserved for downstream use

---

## 🚀 Tips

* Always shift IDs (`shift_id_by=1`) so `0` is used for padding.
* Use `ignore_last_n=1` during training (last item becomes prediction target).
* Ensure lists are well-formatted — e.g., `"[]"` is valid, `"NaN"` or malformed strings will crash.
* You can use the `chronological` flag to switch between reverse-chronology and natural order truncation.

---

## 📂 Optional: MultiFileDatasetV2

If your dataset is too large to fit in memory, split it into multiple CSVs:

```
tmp/mydataset_0.csv
tmp/mydataset_1.csv
...
```

And create a `tmp/mydataset_users.csv` like:

```
0, 1000
1, 1200
2, 500
...
```

This tells the loader how many users are in each shard.

Then, use:

```python
MultiFileDatasetV2(
    file_prefix="tmp/mydataset",
    num_files=16,
    padding_length=51,
    ...
)
```

---

## ✅ Ready to Train?

Once your dataset is ready, set:

```python
train_fn.dataset_name = "my-dataset"
```

And plug your dataset into `get_common_preprocessors()` to return your `.csv` path.

You're all set!

```

---
