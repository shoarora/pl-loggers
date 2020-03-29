# Comparing loggers through pytorch-lightning

## Why pytorch-lightning?
I want to compare a bunch at once, and they've done a great
job of standardizing the logger wrappers and interfaces.
It's really easy to initialize a different type and
get to work with it.

This does mean that I don't learn about what it takes
to integrate the logger code into the training loop.
But I'm personally okay with operating under the assumption
that this step is similar across loggers.  This lets me focus on...

## What I care about in comparing:

- managing and tracking experiment hyperparams
- comparing results across runs
- checkpointing, resuming training
- usable in both training clusters and notebooks/colab type settings

## Tensorboard
```sh
python train.py --save_dir='tensorboard' --name='tensorboard-0' --version='0' --logger_type='tensorboard'
```

```
tensorboard --logdir tensorboard
```

## CometML
```sh
python train.py --save_dir='comet' --name='comet-0' --logger_type comet --api_key ce5Sj9A2Mko6heJH5FOjBQxet

...
COMET INFO: Experiment is live on comet.ml https://www.comet.ml/shoarora/pl-loggers/a51199ef5824451897f3ab0c1900ec4b
```

## MLFlow
```sh
# run local
python train.py --name='mlflow-0' --logger_type mlflow --tracking_uri file:/Users/shoarora/Developer/pl-loggers/mlflow
```
```
sh
# run on managed mlflow
python train.py --name='/Users/shoarora@mercari.com/pl-loggers' --logger_type mlflow --tracking_uri databricks:/Users/shoarora@mercari.com/pl-loggers
```
This appears to not work with checkpointing since it tries to create a weird save path.  The feature set feels largely the same though.

## Neptune
```sh
python train.py --logger_type neptune --api_key $NEPTUNE_API_TOKEN --name shoarora/pl-loggers
```

## Test tube
```sh
python train.py --save_dir='test_tube' --version=0 --logger_type test_tube
```

## Weights and biases
```sh
python train.py --save_dir='wandb' --name wandb-0 --logger_type wandb
```

## TRAINS
```sh
python train.py --logger_type trains --name='trains-0'
```

## My conclusion
