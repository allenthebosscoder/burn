# Backend

## `Backend`

- Specifies how tensors and operations are implemented.
- Allows the same model code to run on CPU, WGPU, CUDA, etc.

## `AutodiffBackend`

- Extends `Backend`.
- Supports gradient computation.
- Enables `loss.backward()`.

--- 
# Tensor

## Generic Arguments

The `Tensor` struct has 3 generic arguments:

```rust
Tensor<B, D, K>
```

- `B` = backend
- `D` = dimensionality
- `K` = data type / kind

Common forms:

```rust
Tensor<B, D>           // Float tensor by default
Tensor<B, D, Float>    // Explicit float tensor
Tensor<B, D, Int>      // Int tensor
Tensor<B, D, Bool>     // Bool tensor
```

The exact element types depend on the backend implementation.

Examples:

```text
Float -> f32, f64
Int   -> i32, i64
Bool  -> bool
```

`dtype` means the data type stored in the tensor.

---

## Dimensionality vs Shape

Burn tensors are declared using their number of dimensions `D`.

`D` is not the number of elements.

`D` is also not the full shape.

Example:

```rust
let floats = [1.0, 2.0, 3.0, 4.0, 5.0];

let device = Default::default();

let tensor_1 = Tensor::<Backend, 1>::from_floats(floats, &device);
```

This is correct because:

- The tensor is 1-dimensional.
- The shape is `[5]`.
- The tensor has 5 elements.

This would be incorrect:

```rust
let tensor_1 = Tensor::<Backend, 5>::from_floats(floats, &device);
```

because `Tensor::<Backend, 5>` means a 5-dimensional tensor, not a tensor with 5 elements.

---

## Examples of Dimensions and Shape

### 1-D Tensor

```rust
Tensor<Backend, 1>
```

Example contents:

```text
[1, 2, 3, 4, 5]
```

Shape:

```text
[5]
```

Meaning:

- 1 dimension
- 5 elements

---

### 2-D Tensor

```rust
Tensor<Backend, 2>
```

Example contents:

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

Shape:

```text
[2, 3]
```

Meaning:

- 2 dimensions
- 2 rows
- 3 columns
- 6 total elements

---

### 3-D Tensor

```rust
Tensor<Backend, 3>
```

Example contents:

```text
[
  [
    [1, 2],
    [3, 4]
  ],
  [
    [5, 6],
    [7, 8]
  ]
]
```

Shape:

```text
[2, 2, 2]
```

Meaning:

- 3 dimensions
- 2 blocks
- 2 rows per block
- 2 columns per row
- 8 total elements

---

## Initialization

Burn tensors are commonly initialized using:

```rust
Tensor::from_data(...)
```

The `from_data()` method takes data and places it into a tensor on a specific device.

Example:

```rust
let device = Default::default();

let tensor = Tensor::<Backend, 1>::from_data([1.0, 2.0, 3.0], &device);
```

---

## TensorData

`TensorData` stores:

- shape
- dtype
- values

Example:

```text
TensorData {
    shape: [3],
    dtype: Float,
    values: [1.0, 2.0, 3.0]
}
```

This completely describes a tensor's contents.

Example:

```rust
let data = TensorData::from([1.0, 2.0, 3.0]);
```

This creates tensor contents, but does not place them on a device.

To create a Burn tensor:

```rust
let tensor =
    Tensor::<Backend, 1>::from_data(
        data,
        &device,
    );
```

As a Burn user, you will mostly work with:

```rust
Tensor
```

rather than:

```rust
TensorData
```

`TensorData` is mainly used when creating tensors or extracting tensor contents.

---

## from_floats

`from_floats()` is a convenience method for float tensors.

Example:

```rust
let tensor = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);
```

This is recommended for float initialization.

Internally, Burn converts the float array into `TensorData`.

---

## Int Tensor Example

Use `Int` when the tensor stores integer values.

Example:

```rust
let arr: [i32; 6] = [1, 2, 3, 4, 5, 6];

let tensor = Tensor::<Backend, 1, Int>::from_data(
    TensorData::from(&arr[0..3]),
    &device,
);
```

This creates an integer tensor from:

```text
[1, 2, 3]
```

Shape:

```text
[3]
```

---

## Custom Type Example

Custom Rust data can be converted into tensor data manually.

Example:

```rust
struct BodyMetrics {
    age: i8,
    height: i16,
    weight: f32,
}

let bmi = BodyMetrics {
    age: 25,
    height: 180,
    weight: 80.0,
};

let data = TensorData::from([
    bmi.age as f32,
    bmi.height as f32,
    bmi.weight,
]);

let tensor = Tensor::<Backend, 1>::from_data(data, &device);
```

This creates:

```text
[25.0, 180.0, 80.0]
```

Shape:

```text
[3]
```

---

## Getting Data Back

Use:

```rust
tensor.to_data()
```

when you want to retrieve the data but still reuse the tensor afterward.

Use:

```rust
tensor.into_data()
```

when you only need the data once and will not reuse the tensor afterward.

---

## Important Takeaway

```rust
Tensor<Backend, 1>
```

means:

```text
1-dimensional tensor
```

not:

```text
tensor with 1 element
```

Examples:

```text
[5]
[100]
[1000]
```

are all valid shapes for:

```rust
Tensor<Backend, 1>
```

because they all have exactly one dimension.

## Ownership

Most Burn tensor operations take ownership of the input tensor.

Example:

```rust
let min = input.min();
```

After this call:

```text
input
```

has been moved.

It can no longer be used unless it was cloned beforehand.

This follows Rust ownership rules.

---

## Why Cloning is Needed

Example:

```rust
let input = Tensor::<Wgpu, 1>::from_floats(
    [1.0, 2.0, 3.0, 4.0],
    &device,
);

let min = input.clone().min();
let max = input.clone().max();

let input =
    (input.clone() - min.clone())
        .div(max - min);
```

Cloning is necessary because:

- `min()` takes ownership of the tensor.
- `max()` takes ownership of the tensor.
- Arithmetic operations also take ownership.

Without cloning:

```rust
let min = input.min();
let max = input.max(); // Error
```

because `input` was already moved into `min()`.

---

## Clone Does NOT Copy Tensor Data

Burn tensors do not implement:

```rust
Copy
```

so cloning must be explicit.

However:

```rust
tensor.clone()
```

does NOT duplicate the tensor buffer.

Instead:

```text
- Reuses the same underlying tensor data
- Increases a reference count
```

Therefore cloning is cheap.

---

## Example

```rust
let input = Tensor::<Wgpu, 1>::from_floats(
    [1.0, 2.0, 3.0, 4.0],
    &device,
);

let min = input.clone().min();
let max = input.clone().max();

let input =
    (input.clone() - min.clone())
        .div(max - min);
```

Result:

```text
[0.0, 0.33333334, 0.6666667, 1.0]
```

This performs min-max normalization:

```text
(x - min) / (max - min)
```

---

## Moving Still Applies

Example:

```rust
let input =
    (input.clone() - min.clone())
        .div(max - min);
```

Notice:

```rust
max - min
```

moves:

```text
max
min
```

into the subtraction operation.

Afterward:

```rust
println!("{:?}", min.to_data());
```

would fail.

To use them again:

```rust
max.clone()
min.clone()
```

would be required.

---

## Why Burn Uses Ownership

Because Burn knows exactly how many times a tensor is used.

This allows optimizations such as:

- Tensor buffer reuse
- Kernel fusion (`burn-fusion`)
- Automatic inplace operations

---

## Inplace Operations

Burn does not expose explicit inplace operations.

Instead:

```text
If a tensor has only one owner:
    Burn can safely perform inplace operations internally.
```

This allows:

- Cleaner API
- Better optimization opportunities
- No manual inplace management by the user

---

## Important Takeaway

```rust
tensor.clone()
```

does NOT mean:

```text
copy all tensor data
```

It means:

```text
create another owner of the same tensor data
```

Cloning is primarily needed to satisfy Rust ownership rules while allowing a tensor to be reused multiple times.

---

# Autodiff

Autodiff (automatic differentiation) is Burn's mechanism for computing gradients during training.

Without autodiff:

```rust
loss.backward()
```

cannot be performed.

---

## `AutodiffBackend`

Burn separates:

```rust
Backend
```

from:

```rust
AutodiffBackend
```

`Backend` provides:

- tensor storage
- tensor operations
- device execution

`AutodiffBackend` extends `Backend` and adds:

- gradient computation
- backward propagation
- computational graph tracking

This allows Burn to avoid gradient-tracking overhead during inference.

---

## Adding Autodiff Support

Burn provides an autodiff wrapper:

```rust
type TrainingBackend = Autodiff<Wgpu>;
```

This wraps an existing backend and adds gradient computation support.

The underlying backend still executes tensor operations.

The autodiff layer only tracks information needed for gradient computation.

---

## Computational Graph

When tensors participate in differentiable operations, Burn records how they were produced.

Example:

```rust
let y = x.clone() * 3.0;
let z = y.clone() + 1.0;
```

Conceptually:

```text
x
|
*3
|
y
|
+1
|
z
```

Calling:

```rust
z.backward()
```

traverses this graph in reverse and computes gradients.

---

## `B::Gradients`

Unlike PyTorch, gradients are not stored directly on tensors.

PyTorch:

```python
loss.backward()

tensor.grad
```

Burn:

```rust
let gradients = loss.backward();
```

returns a separate gradient container.

Type:

```rust
B::Gradients
```

Gradients can then be retrieved from the container for a specific tensor.

This design works better with Rust ownership and makes gradients explicit.

---

## Training vs Inference

Training typically uses:

```rust
type B = Autodiff<Wgpu>;
```

Inference typically uses:

```rust
type B = Wgpu;
```

Since inference does not require gradients, there is no reason to pay the cost of building a computational graph.

Unlike PyTorch:

```python
torch.no_grad()
torch.inference_mode()
```

Burn usually separates training and inference at the backend type level.

---

## Validation

When using an autodiff backend, the underlying tensor can be extracted with:

```rust
tensor.inner()
```

This returns a tensor using the backend's inner backend and removes autodiff tracking.

This is useful during validation when gradients are unnecessary.

---

## Important Takeaway

Burn treats gradient computation as an optional layer on top of a backend.

```rust
Autodiff<MyBackend>
```

adds:

- computational graph tracking
- gradient computation
- `backward()`

Unlike PyTorch, gradients are stored in a separate container returned by `backward()` rather than directly on tensors.

---

# Module

## What is a Module?

A module is Burn's equivalent of a PyTorch:

```python
nn.Module
```

A module is a reusable neural network component.

Examples:

- an entire model
- a convolution block
- a transformer layer
- a feedforward network

Modules can contain other modules, allowing large models to be built from smaller pieces.

Example:

```rust
#[derive(Module, Debug)]
pub struct Model {
    conv1: Conv2d,
    conv2: Conv2d,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear,
    linear2: Linear,
    activation: Relu,
}
```

Each field is itself a module.

Together they form a larger module.

---

## Why Does Burn Need `#[derive(Module)]`?

Suppose we have:

```rust
pub struct Model {
    conv1: Conv2d,
    conv2: Conv2d,
    linear1: Linear,
}
```

Rust sees:

```text
A struct with three fields
```

but Burn sees nothing special.

Burn does not know:

- where parameters are stored
- which tensors should be optimized
- which tensors should be saved

Adding:

```rust
#[derive(Module)]
```

teaches Burn how to recursively walk through the struct and find all parameters.

Conceptually:

```text
Model
├── Conv2d
│   ├── weight
│   └── bias
├── Conv2d
│   ├── weight
│   └── bias
└── Linear
    ├── weight
    └── bias
```

This allows Burn to automatically discover every parameter in the model.

---

## Modules Form a Tree

Modules can contain other modules.

Example:

```text
Model
├── Encoder
│   ├── Attention
│   ├── Attention
│   └── FeedForward
└── Classifier
```

Each module is responsible for its own parameters.

Burn recursively walks the tree and collects all parameters from all child modules.

This is why operations such as:

```rust
model.to_device(...)
optimizer.step(...)
```

can work on an entire model rather than individual tensors.

---

## Forward Pass

The Module derive does not generate a forward function.

You must implement it yourself.

Example:

```rust
impl Model {
    pub fn forward(
        &self,
        images: Tensor<3>,
    ) -> Tensor<2> {
        ...
    }
}
```

The forward pass defines how data flows through the module.

The Module derive only handles parameter management.

---

## What is a Parameter?

A parameter is a tensor that should be learned during training.

Examples:

```text
Linear weights
Linear bias
Convolution kernels
Embedding matrices
```

During training:

```text
forward
    ↓
loss
    ↓
backward
    ↓
optimizer updates parameters
```

Only parameters are updated by the optimizer.

---

## `Param<Tensor>` vs `Tensor`

Most built-in modules already manage their own parameters internally.

Examples:

```rust
Linear
Conv2d
Embedding
```

already contain trainable weights.

You only need:

```rust
Param<Tensor<B, D>>
```

when creating your own module that directly stores tensors.

Example:

```rust
#[derive(Module)]
pub struct MyLayer<B: Backend> {
    weight: Param<Tensor<B, 2>>,
}
```

The wrapper tells Burn:

```text
This tensor is a parameter.
```

Burn will then:

- assign it a parameter ID
- include it in parameter traversal
- allow optimizers to update it

---

### Plain Tensor

```rust
Tensor<B, D>
```

is just data.

Examples:

- constants
- masks
- cached values

Burn does not treat these tensors as parameters.

They are not updated by optimizers.

---

## Important Takeaway

A module is Burn's way of representing a trainable neural network component.

```rust
#[derive(Module)]
```

allows Burn to recursively discover parameters inside a module tree.

Most of the time you will build modules from existing modules such as:

```rust
Linear
Conv2d
Dropout
Embedding
```

and only use:

```rust
Param<Tensor<B, D>>
```

when creating your own trainable tensors.

---

# Learner

## What is a Learner?

A learner is Burn's high-level training abstraction.

Instead of manually writing:

```text
for epoch
    for batch
        forward
        backward
        optimizer step
        validation
        logging
        checkpointing
```

Burn provides a learner that manages the training workflow automatically.

The learner is provided by the:

```rust
burn-train
```

crate.

---

## Why Use a Learner?

Training a model involves more than just:

```text
forward
backward
optimizer step
```

A complete training loop often also needs:

- validation
- metric tracking
- logging
- checkpointing
- learning rate scheduling
- multi-device support

The learner provides these features so they do not need to be implemented from scratch.

---

## `SupervisedTraining`

The main training abstraction is:

```rust
SupervisedTraining
```

It represents a configurable training loop.

The learner is typically created using:

- a training dataloader
- a validation dataloader

Conceptually:

```text
SupervisedTraining
├── Training DataLoader
└── Validation DataLoader
```

---

## Assumptions

The built-in learner assumes the standard machine learning workflow:

```text
Train on Training Dataset
Validate on Validation Dataset
```

This supports:

- supervised learning
- unsupervised learning
- fine-tuning

For more specialized workflows, a custom training loop may be required.

---

## Configuration

A learner can be configured with:

- metrics
- metric plotting
- logging
- checkpointing
- gradient accumulation
- multiple devices
- number of epochs
- learning rate scheduling

The goal is to make the training loop configurable rather than forcing users to implement these features manually.

---

## Launching Training

Once configured, training is started with:

```rust
learner.launch(...)
```

The learner is given:

- a model
- an optimizer
- a learning rate scheduler (or constant learning rate)

The learner then:

```text
Epoch
├── Train
├── Validate
├── Record Metrics
└── Save Checkpoint
```

When training finishes:

```rust
launch(...)
```

returns the trained model.

---

## Artifacts

During training, the learner automatically saves information to disk.

Example:

```text
experiment.log
checkpoint/
train/
valid/
```

These artifacts may include:

- logs
- metrics
- model checkpoints
- optimizer state
- scheduler state

---

## Checkpoints

A checkpoint stores the current training state.

Typically:

```text
Model Parameters
Optimizer State
Scheduler State
```

This allows training to be resumed later instead of starting from scratch.

Example:

```text
checkpoint/
├── model-1.mpk.gz
├── optim-1.mpk.gz
└── scheduler-1.mpk.gz
```

---

## Metrics

The learner can automatically track metrics during training and validation.

Examples:

- Loss
- Accuracy
- Precision
- Recall

Metrics are covered in the next section.

---

## Important Takeaway

The learner is Burn's high-level training manager.

Instead of manually managing:

- training loops
- validation loops
- logging
- metrics
- checkpoints

Burn provides a configurable learner that manages the training workflow while allowing you to focus on the model itself.

---

# Metric

## What is a Metric?

A metric is a measurement collected during training or validation.

Metrics are used to evaluate how well a model is performing.

Examples:

- Loss
- Accuracy
- Precision
- Recall
- F1 Score
- Perplexity

Metrics are typically registered with a learner and automatically tracked throughout training.

---

## Metrics vs Loss

A loss is used to train the model.

Metrics are used to monitor model performance.

Some values, such as loss, can serve both purposes.

Example:

```text
Loss
    ↓
backward()
    ↓
optimizer updates parameters
```

Metrics are typically used for monitoring rather than parameter updates.

They measure model performance.

For example:

```text
Loss
Accuracy
Precision
Recall
```

may all be computed for the same batch, but only the loss is used during backpropagation.

---

## How Metrics Work

Metrics do not operate directly on model outputs.

Instead, Burn uses an adaptor system.

Conceptually:

```text
Model Output
      ↓
   Adaptor
      ↓
 Metric Input
      ↓
    Metric
```

Each metric defines its own input type.

This allows multiple metrics to operate on the same model output.

---

## Adaptors

An adaptor converts a model output into the input expected by a metric.

Example:

```text
ClassificationOutput
        ↓
Adaptor
        ↓
AccuracyInput
        ↓
Accuracy Metric
```

The metric only knows how to work with:

```rust
AccuracyInput
```

The adaptor handles the conversion.

---

## Built-in Output Types

Burn provides several common output types.

### `ClassificationOutput`

Used for:

```text
Single-label classification
```

Examples:

- digit classification
- image classification
- sentiment analysis

---

### `MultiLabelClassificationOutput`

Used for:

```text
Multi-label classification
```

Examples:

- image tagging
- multi-label prediction

---

### `RegressionOutput`

Used for:

```text
Regression tasks
```

Examples:

- house price prediction
- temperature prediction

---

### `SequenceOutput`

Used for:

```text
Sequence prediction
```

Examples:

- language modeling
- speech recognition
- text generation

---

## Built-in Adaptors

These built-in output types already implement adaptors for many common metrics.

For example:

```text
ClassificationOutput
├── Accuracy
├── Precision
├── Recall
├── F-Beta Score
├── AUROC
└── Loss
```

As a result, most metrics work automatically without requiring additional code.

---

## Custom Adaptors

If a metric cannot work with your output type, you can implement:

```rust
Adaptor<T>
```

yourself.

Example:

```rust
impl<B: Backend>
    Adaptor<AccuracyInput<B>>
    for ClassificationOutput<B>
{
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(
            self.output.clone(),
            self.targets.clone(),
        )
    }
}
```

This tells Burn how to convert:

```text
ClassificationOutput
```

into:

```text
AccuracyInput
```

---

## Custom Metrics

Custom metrics are created by implementing:

```rust
Metric
```

A metric:

- receives inputs
- updates internal state
- produces metric values

Examples:

- custom evaluation metrics
- domain-specific metrics
- hardware monitoring metrics

---

## Numeric Metrics

Many metrics produce numeric values.

Examples:

```text
Loss
Accuracy
Precision
Recall
```

Numeric metrics can additionally implement:

```rust
Numeric
```

This allows Burn to:

- plot metrics
- track metric history
- visualize training progress

---

## Important Takeaway

Metrics are separate from model outputs.

Burn uses:

```text
Output
   ↓
Adaptor
   ↓
Metric Input
   ↓
Metric
```

to allow many different metrics to operate on the same output type.

Most common tasks already have built-in output types and adaptors, so metrics usually work automatically.

---

# Config

## What is a Config?

A config stores the hyperparameters used to construct a module.

Examples:

- hidden size
- dropout rate
- number of layers
- number of classes

Example:

```rust
#[derive(Config)]
pub struct MyModuleConfig {
    d_model: usize,
    d_ff: usize,

    #[config(default = 0.1)]
    dropout: f64,
}
```

This config does not contain any weights.

It only stores the values needed to create a module.

Conceptually:

```text
d_model = 512
d_ff = 2048
dropout = 0.1
```

---

## Why Use Configs?

Machine learning models often have many hyperparameters.

Without configs, these values would need to be manually passed around whenever a model is created.

Configs provide:

- a single place for model configuration
- default values
- serialization
- reproducibility

---

## `#[derive(Config)]`

The Config derive automatically provides:

- a constructor
- default values
- serialization support
- builder-style methods

Example:

```rust
let config =
    MyModuleConfig::new(512, 2048);
```

Fields marked with:

```rust
#[config(default = 0.1)]
```

automatically receive default values.

So:

```rust
let config =
    MyModuleConfig::new(512, 2048);
```

creates:

```text
d_model = 512
d_ff = 2048
dropout = 0.1
```

---

## Builder Methods

The Config derive automatically generates:

```rust
with_<field>()
```

methods.

Example:

```rust
let config =
    MyModuleConfig::new(512, 2048);

let config =
    config.with_dropout(0.2);
```

The returned config contains:

```text
d_model = 512
d_ff = 2048
dropout = 0.2
```

This is similar to the builder pattern.

---

## Config vs Module

A config and a module are different types.

Example:

```rust
#[derive(Config)]
pub struct ModelConfig {
    hidden_size: usize,
    dropout: f64,
}

#[derive(Module)]
pub struct Model {
    linear: Linear,
}
```

The config stores hyperparameters.

```text
hidden_size
dropout
```

The module stores parameters.

```text
weights
biases
```

The config does not contain the module.

The module does not contain the config.

---

## `init()`

A common Burn pattern is to implement:

```rust
init(...)
```

on the config type.

Example:

```rust
impl ModelConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Model {
        ...
    }
}
```

The purpose of:

```rust
init()
```

is to create a module using the values stored in the config.

Conceptually:

```text
ModelConfig
(hidden_size = 512)

        |
        | used by
        v

config.init()

        |
        v

Model
(weights, biases, ...)
```

The config provides the information needed to construct the module.

The module contains the actual initialized parameters.

---

## Typical Workflow

Burn code commonly follows this pattern:

```rust
let config =
    ModelConfig::new(...);

let model =
    config.init::<Wgpu>(&device);

let output =
    model.forward(input);
```

Conceptually:

```text
Config
    ↓
init()
    ↓
Module
    ↓
forward()
    ↓
Output
```

More precisely:

```text
Config
    stores hyperparameters

init()
    creates parameters

Module
    stores parameters

forward()
    uses parameters to process data
```

---

## Serialization

Configs can be saved and loaded.

Example:

```rust
config.save("config.json").unwrap();
```

This allows experiment configurations to be stored and reused later.

---

## Important Takeaway

A config is a blueprint for constructing a module.

A module is the actual neural network component containing trainable parameters.

```text
Config
    ↓
init()
    ↓
Module
```

means:

```text
Use the config to create the module.
```

not:

```text
The config becomes the module.
```

`#[derive(Config)]` provides:

- constructors
- default values
- serialization
- builder methods

while:

```rust
#[derive(Module)]
```

creates a trainable neural network component that stores parameters.




# Record

## What is a Record?

A record is Burn's representation of a module's state.

Conceptually, a record is similar to a PyTorch:

```python
state_dict
```

A record contains the information needed to restore a module's parameters.

Example:

```text
Linear
├── weight
└── bias
```

can be converted into:

```text
LinearRecord
├── weight
└── bias
```

The record stores the state.

The module stores the behavior.

---

## Why Does Burn Use Records?

Modules cannot be directly serialized.

A module may contain:

- parameters
- devices
- backend-specific types
- non-serializable fields

Example:

```rust
#[derive(Module)]
pub struct Model {
    linear: Linear,
}
```

Burn cannot simply write the entire module to disk.

Instead, Burn extracts the module's state into a record.

Conceptually:

```text
Module
    ↓
into_record()
    ↓
Record
    ↓
Save
```

and later:

```text
Record
    ↓
Load
    ↓
Module
```

This separates:

```text
Model Structure
```

from:

```text
Model State
```

---

## Module vs Record

A module contains:

- parameters
- methods
- forward pass logic

Example:

```rust
model.forward(input)
```

A record contains:

- parameter values

only.

Records do not contain:

- forward functions
- training logic
- methods

Conceptually:

```text
Module
├── Parameters
├── Methods
└── Forward Pass

Record
└── Parameters
```

---

## Records and Configs

Configs, modules, and records all serve different purposes.

```text
Config
    ↓
Creates
    ↓
Module
    ↓
Produces
    ↓
Record
```

Responsibilities:

```text
Config
    -> stores hyperparameters

Module
    -> stores parameters and behavior

Record
    -> stores parameter values
```

---

## Saving a Record

A module can be converted into a record:

```rust
module.into_record()
```

The record can then be saved using a recorder.

Conceptually:

```text
Model
    ↓
Record
    ↓
File
```

---

## Loading a Record

A saved record can be loaded back into a module.

Conceptually:

```text
File
    ↓
Record
    ↓
Module
```

This restores the parameter values without recreating the model architecture.

The architecture must already exist.

---

## Recorders

A recorder defines:

- how records are serialized
- where records are stored

Examples:

```text
MessagePack
Binary
JSON
```

Some recorders:

- prioritize speed
- prioritize file size
- prioritize readability

Burn provides multiple recorder implementations for different use cases.

---

## Precision Conversion

Records are independent of the precision used during training.

For example:

```text
Train with:
f32
```

and save as:

```text
f16
```

or:

```text
f64
```

Burn automatically performs the necessary conversions when saving and loading.

---

## Backend Independence

Records are independent of the backend.

Example:

```text
Train on:
Wgpu

Save Record

Load on:
NdArray
```

The record stores parameter values rather than backend-specific tensor implementations.

This makes model portability easier.

---

## Why Not Serialize the Whole Module?

Burn intentionally separates:

```text
Module Structure
```

from:

```text
Module State
```

This provides:

- backend independence
- automatic precision conversion
- serialization safety
- support for non-serializable module fields

Instead of saving:

```text
Entire Model
```

Burn saves:

```text
Model State
```

through records.

---

## Important Takeaway

A record is Burn's version of a model state.

Conceptually:

```text
Config
    ↓
creates
    ↓
Module
    ↓
produces
    ↓
Record
```

Where:

```text
Config
    -> hyperparameters

Module
    -> parameters + behavior

Record
    -> parameter values only
```

Records are used for saving and loading model state independently of the backend being used.

---

# Dataset

## What is a Dataset?

A dataset is a collection of items used for training or evaluation.

Examples:

- images
- text
- audio
- video

A dataset is responsible for retrieving data from some source.

Examples:

- memory
- SQLite database
- CSV file
- Hugging Face dataset

---

## The `Dataset` Trait

Burn represents datasets using the:

```rust
Dataset
```

trait.

```rust
pub trait Dataset<I>: Send + Sync {
    fn get(&self, index: usize) -> Option<I>;
    fn len(&self) -> usize;
}
```

A dataset:

- has a fixed length
- supports random access
- returns items by index

Conceptually:

```text
Dataset
├── Item 0
├── Item 1
├── Item 2
└── ...
```

---

## Dataset Items

A dataset stores items.

The item type depends on the problem.

Example:

```rust
pub struct MnistItem {
    pub image: [[f32; 28]; 28],
    pub label: u8,
}
```

Each call to:

```rust
dataset.get(index)
```

returns one item.

---

## Dataset vs Tensor

A dataset is not a tensor.

A dataset stores individual samples.

Example:

```text
Image + Label
Image + Label
Image + Label
...
```

A tensor batch is created later during data loading.

---

## Data Loading Pipeline

During training, data typically flows through:

```text
Dataset
    ↓
DataLoader
    ↓
Batcher
    ↓
Tensor Batch
    ↓
Model
```

Responsibilities:

```text
Dataset
    -> retrieves samples

DataLoader
    -> loads samples

Batcher
    -> combines samples into batches

Model
    -> processes batches
```

---

## Why Use a Batcher?

Models usually process batches rather than individual samples.

Example:

```text
Dataset Item
    Image
    Label
```

becomes:

```text
Batch
├── Images Tensor
└── Labels Tensor
```

The batcher is responsible for this conversion.

---

## Dataset Transformations

Burn provides dataset transformations.

A transformation wraps another dataset and modifies its behavior.

Conceptually:

```text
Dataset
    ↓
Transformation
    ↓
New Dataset
```

Transformations are lazy.

This means items are transformed only when requested.

No preprocessing is performed ahead of time.

---

## Common Transformations

### `MapperDataset`

Applies a transformation to every item.

Example:

```text
Raw Bytes
    ↓
Image
```

Useful for:

- normalization
- parsing
- format conversion

---

### `ShuffledDataset`

Returns dataset items in shuffled order.

Useful before creating train/validation splits.

---

### `PartialDataset`

Returns only a portion of a dataset.

Useful for:

- train splits
- validation splits
- test splits

Example:

```text
Dataset
├── First 80%  -> Train
└── Last 20%   -> Test
```

---

### `ComposedDataset`

Combines multiple datasets into a single dataset.

Conceptually:

```text
Dataset A
      +
Dataset B
      ↓
Combined Dataset
```

---

### `WindowsDataset`

Creates overlapping windows from sequential data.

Useful for:

- time series
- sequence models
- LSTMs

---

## Dataset Storage

Burn provides multiple storage implementations.

### `InMemDataset`

Stores all items in memory.

Best for:

```text
Small datasets
```

---

### `SqliteDataset`

Stores items in SQLite.

Best for:

```text
Large datasets
```

---

### `DataframeDataset`

Stores items in a Polars dataframe.

Best for:

```text
Data analysis
Tabular data
```

---

## Hugging Face Datasets

Burn can load datasets from Hugging Face.

Example:

```rust
let dataset: SqliteDataset<DbPediaItem> =
    HuggingfaceDatasetLoader::new("dbpedia_14")
        .dataset("train")
        .unwrap();
```

Downloaded datasets are typically stored using SQLite.

---

## No Streaming Dataset API

Burn intentionally does not provide a streaming dataset API.

Datasets are expected to provide:

```rust
len()
get(index)
```

The learner uses the dataset length to determine:

- epoch size
- validation frequency
- checkpoint frequency

---

## Important Takeaway

A dataset is a collection of items that can be accessed by index.

During training:

```text
Dataset
    ↓
DataLoader
    ↓
Batcher
    ↓
Tensor Batch
    ↓
Model
```

Burn keeps datasets simple and uses lazy transformations to build more complex data pipelines.