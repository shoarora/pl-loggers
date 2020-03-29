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
python train.py --save_dir='comet' --name='comet-0' --logger_type comet --api_key <api-key>

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

## Multiple
Lightning let's you pass a list of loggers if for whatever reason you want to support multiple of these at once.


## My conclusion
This is a pretty non-exhaustive look at all the features, but it's enough for me personally to take my pick.

Tensorboard is the baseline here.  It let's me plot all my runs, but requires all the data to be hosted by me.
Not the best for granular analysis.  Test-tube builds on tensorboard with hyperparameter search features,
but they're centered largely around SLURM, which I don't use.

TRAINS and MLFlow can be self-deployed on-premise for free, which is super cool.  MLFlow also has features
around managing environments, experiments and deploying model servers.  But I didn't find the UIs particularly pleasant to use.
They're feature-sufficient, but not _enjoyable_.  This wouldn't be a problem if the hosted loggers weren't so nice.
I wish I liked one of these, since that would make it really easy for me to use the same platform for work and personal use at no cost.

Comet, Neptune, and W&B are the hosted platforms.  They all offer enterprise solutions that can be deployed on-premises (not for free).
Right off the bat, Neptune had the slowest UI, pages took 1.5s to load.  This is fair since there's a lot of data in ML, but the other
ones do it faster.

I don't think I can go wrong with either comet or W&B.  Ultimately, I'm going to start with Comet, and see where that takes me.
If I'm being honest, the corporate names listed on their [home page](https://www.comet.ml/site/) is influencing my decision a bit.

All of these are pretty feature-rich (here's a [comparison chart](https://neptune.ai/blog/best-ml-experiment-tracking-tools)
 created by Neptune, maybe biased but also thorough), so the decision is
largely coming down to UI/UX.  That's not something I could easily capture
in writing.  If you're interested, I encourage you to check them out yourself.  This repo integrates pretty easily into
all of them so it should be easy to log your first experiment and play around.
