use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
}

#[derive(Clone, Default)]
pub struct MnistBatcher {}

#[derive(Clone, Debug)]
pub struct MnistBatch {
    pub images: Tensor<3>,
    pub targets: Tensor<1, Int>,
}

impl Batcher<MnistItem, MnistBatch> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &Device) -> MnistBatch {
        let images = items                                              // take items Vec<MnistItem>
            .iter()                                                     // create an iterator over it
            .map(|item| TensorData::rom(item.image))                    // for each item, convert the image to float data struct
            .map(|data| Tensor::<2>::from_data(data, device))           // for each data struct, create a tensor on the device
            .map(|tensor| tensor.reshape([1, 28, 28]))                  // for each tensor, reshape to the image dimensions [C, H, W]
            // Normalize: scale between [0,1] and make the mean=0 and std=1
            // values mean = 0.1307, std = 0.3081 are from the PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)           // for each tensor, apply normalization
            .collect();                                                 // consume the resulting iterator & collect the values into a new vector

        let targets = items
            .iter()
            .map(|item| Tensor::<1, Int>::from_data([item.label as i64], device))
            .collect();
        
        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(images, 0);

        MnistBatch { images, targets }
    }
}