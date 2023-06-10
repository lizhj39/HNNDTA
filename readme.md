## file list

- data_preprocess_multi_condi.py：Preprocessing operations were performed on the raw data from ChEMBL.

- model_train.py：Train the HNNDTA model

- model_test.py：Run the HNNDTA model on test dataset and get predicted DTA value from test dataset.

- predict_FDA.py: Run the HNNDTA model on FDA candidate drug dataset and rank all drug based on the DTA value.
  

## run code

To train the HNNDTA model:

```python
python model_train.py
```

To run the HNNDTA model on test dataset:

```python
python model_test.py
```



## requirement

dgl
dglgo
numpy
pandas
prettytable
rdkit
torch

