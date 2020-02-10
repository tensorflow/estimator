# Spatial partitioning

Spatial partitioning allows us to run models with larger input images. Typically
these models will be too large to fit on a single TPU core.

Spatial partitioning uses multiple cores to process different parts of the input
tensor. Each core communicates with the other cores when necessary to merge
overlapping parts of the computation. All the complicated merging logic is
implemented in the XLA compiler, therefore you only need to configure how the
inputs to your model are partitioned.

Note: Spatial partitioning only distributes activations across multiple cores.
Each core still maintains its own copy of the model weights. For most image
model, activations use more memory than the model weights.

## Enabling Spatial Partitioning with TPUEstimator

Spatial partitioning doesn't require any code change in your model. You only 
need to specify the spatial partition parameters in your TPUConfig.

```
tpu_config=tpu_config.TPUConfig(
    iterations_per_loop=100,
    num_cores_per_replica=4,
    per_host_input_for_training=tpu_config.InputPipelineConfig.PER_HOST_V2,
    input_partition_dims=[[1, 4, 1, 1], None]]

```

`per_host_input_for_training` must be set to PER_HOST_V2 for spatial
partitioning: this means you must have a tf.data based input pipeline.
`num_cores_per_replica` determines the maximum number partitions we can split.
`input_partition_dims` is a list with two elements: `feature_partition_dims` and
`label_partition_dims` describes how to partition the input tensors. The
structure of `feature_partition_dims` and `label_partition_dims` must match the
structure of features and labels from input_fn.

### Partitioning when features and labels are single tensors

`features` or `labels` can be a single tensor. In this case,
`feature_partition_dims` or `label_partition_dims` must be a list/tuple of
integers or None. The length of the list/tuple must equal to the number of
dimensions of the tensor. For example, if `features` is an image tensor with
shape [N, H, W, C], the `feature_partition_dims` must be a list/tuple with 4
integers.

```
features = image_tensor # [N, H, W, C]
labels = class_label # [N]

input_partition_dims = [[1,4,1,1], None]

```

### Partitioning when features or labels are a dictionary

`features` or `labels` can alternatively be a dictionary from `feature_name` to
a `Tensor`. In this case `feature_partition_dims` or `label_partition_dims` must
be a dict with exactly the same keys, and the value is a list/tuple of integers
or None.

```
features = {'image': image_tensor, 'image_mask': mask_tensor}
labels =  {'class_label': class_id, 'mask': mask_id}

input_partition_dims = [
   {'image': [1,4,1,1], 'image_mask': [1, 2, 2,1]},
   {'class_label': [1], mask: None}]

```

In this example, both `features` and `labels` are dictionaries. Therefore the
`input_partition_dims` contains two dicts with the same structure: the first
dict in `input_partition_dims` has two keys ‘image’ and ‘image_mask’ to match
the tensors in features. The value is a list of integers describes how to
partition the tensor. 'class_label': [1] means we send the class_label tensor to
core 0 only.

### Partitioning when features are a dict, labels are a single tensor

`features` and `labels` could be any of the aforementation’s format. The rule
for `feature_partition_dims` and `label_partition_dims` are applied separately.

```
features = {'image': image_tensor, 'image_mask': mask_tensor}
labels =  class_label # [N]

input_partition_dims = [
   {'image': [1,4,1,1], 'image_mask': [1, 2, 2,1]},
   [1]]

```
