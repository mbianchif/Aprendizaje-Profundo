# Report

## Generate training session results

By executing the `train.py` script, it will execute different training sessions with the given variable parameters.

The constant parameters are:

```py
{
  "dataset": get_mnist_dataset(),
  "optimizer": Adam(
      lr=0.01,
      b1=0.9,
      b2=0.999,
      eps=1e-8,
  ),
  "loss_fn": CrossEntropy(),
  "max_epochs": 48,
  "batch_size": 64,
  "serializer": SparseSerializer(r=0.5),

  # For Parameter Server.
  "sync": BarrierSync(),
  "store": BlockingStore(),
}
```

The variable parameters per training session are:

```py
{
  "addrs": addrs,
  "offline_epochs": offline_epochs,
  "seed": seed,
  "nservers": len(addrs) // 2,
}
```

## Args

| flag | long-flag        | type | multiple | defualt  |
|------|------------------|------|----------|----------|
| -n   | --nodes          | int  | true     | required |
| -e   | --offline_epochs | int  | true     | required |
| -a   | --algorithm      | str  | true     | required |
| -r   | --repeats        | int  | false    | required |
| -s   | --seed           | int  | false    | 67       |

The available algorithm variants are: `["all_reduce", "parameter_server"]`.

Example: `uv run train.py -n 2 4 -e 0 2 4 -a all_reduce parameter_server -r 1`
