#![recursion_limit = "256"]

use burn::tensor::Tensor;
use burn::prelude::*;

fn main() {
    let device = Device::wgpu(DeviceKind::DefaultDevice);
    // Creation of 2 tensors, the first with explicit values and the second with ones,
    // with same shape as the first
    let tensor_1 = Tensor::<2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_2 = Tensor::<2>::ones_like(&tensor_1);

    // Print element-wise addition of the tensors (done in WGPU backend)
    println!("{}", tensor_1 + tensor_2);
}