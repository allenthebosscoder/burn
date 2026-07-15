use burn_core::{
    Tensor,
    module::{Module, ModuleMapper, Param},
};

use crate::{Learner, LearnerModel};
use typing_rules::*;
use macros::fcall;

/// Describes how the module is distributed across multiple devices.
pub struct ModuleSharder;

impl ModuleMapper for ModuleSharder {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = tensor.set_distributed(id);
        Param::from_mapped_value(id, tensor, mapper)
    }
}

impl<M: LearnerModel, L: Label> Learner<M, L> {
    /// Mark the model as sharded across multiple devices.
    pub fn grad_sharded(&mut self) {
        self.model = fcall!(Module::map(Clone::clone(&self.model), &mut (ModuleSharder)));
    }
}
