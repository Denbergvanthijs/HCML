import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def data_preparation(fps, drop=[]):
    # Load csv files
    df_train = pd.read_csv(fps[0])
    df_val = pd.read_csv(fps[1])
    df_test = pd.read_csv(fps[2])

    # Split into X and y
    X_train = df_train.drop(columns=["y"] + drop)
    y_train = df_train["y"]

    X_val = df_val.drop(columns=["y"] + drop)
    y_val = df_val["y"]

    X_test = df_test.drop(columns=["y"] + drop)
    y_test = df_test["y"]

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser(description="process the experiment variables")
    parser.add_argument('--data_path', dest="data_path", type=str,
                        default=Path("./data/default.csv"), help="the path to the data file")
    parser.add_argument('--save_path', dest="save_path", type=str,
                        default=Path("./data/"), help="the path to where we save the files")
    parser.add_argument('--split_sizes', dest="split_sizes", type=str,
                        default="70/15/15", help="the size of the train, validation and test set")
    parser.add_argument('--keep_orig', dest="keep_orig", action='store_true',
                        default=False, help='Whether to also make a stratified split using only sex (does not save the intersectional data)')
    parser.add_argument('--seed', dest="seed", type=int,
                        default=42, help="the random seed")
    args = parser.parse_args()

    # read the data
    df_data = pd.read_csv(args.data_path)

    # Drop the ID column
    # df_data = df_data.drop(columns=["id"])
    # Drop rows where MARRIAGE is not in {1, 2}, ignoring 0 as it should not exist and ignoring 3 which is other
    df_data = df_data[df_data["marriage"].isin([1, 2])]
    print(f"The size of the dataset: {len(df_data)}")

    # Make new column for intersectional group combining sex and marital status
    df_data["intersection"] = (df_data["sex"].astype(str) + df_data["marriage"].astype(str)).astype(int)

    # get the sizes of the different sets
    _, val_size, test_size = tuple(map(lambda x: int(x)/100, args.split_sizes.split("/")))

    # check if we also want to make the stratified split where we keep the sex and marriage columns
    if args.keep_orig:
        # do the sex split
        df_sex = deepcopy(df_data)
        # make the stratified split
        train_data, val_data = train_test_split(df_sex, test_size=val_size + test_size,
                                                random_state=args.seed, stratify=df_data["intersection"])
        # Further split the train_data into train and validation sets
        val_data, test_data = train_test_split(val_data, test_size=test_size / (val_size + test_size),
                                               random_state=args.seed, stratify=val_data["intersection"])

        # drop the intersectional column
        train_data = train_data.drop(columns=["intersection"])
        val_data = val_data.drop(columns=["intersection"])
        test_data = test_data.drop(columns=["intersection"])

        # Save the train, validation, and test sets to CSV files
        train_data.to_csv(Path(f"{args.save_path}/orig_train.csv"), index=False)
        val_data.to_csv(Path(f"{args.save_path}/orig_val.csv"), index=False)
        test_data.to_csv(Path(f"{args.save_path}/orig_test.csv"), index=False)

    # make the stratified split on the intersectional group
    # drop both sex and marriage columns
    df_data = df_data.drop(columns=["sex", "marriage"])

    # do the stratified split
    train_data, val_data = train_test_split(df_data, test_size=val_size + test_size,
                                            random_state=args.seed, stratify=df_data["intersection"])
    # Further split the train_data into train and validation sets
    val_data, test_data = train_test_split(val_data, test_size=test_size / (val_size + test_size),
                                           random_state=args.seed, stratify=val_data["intersection"])

    # Save the train, validation, and test sets to CSV files
    train_data.to_csv(Path(f"{args.save_path}/intersectional_train.csv"), index=False)
    val_data.to_csv(Path(f"{args.save_path}/intersectional_val.csv"), index=False)
    test_data.to_csv(Path(f"{args.save_path}/intersectional_test.csv"), index=False)


if __name__ == "__main__":
    main()
