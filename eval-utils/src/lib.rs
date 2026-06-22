#![allow(warnings)]
use clap::ValueEnum;

pub mod cost_model;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum ComputationType {
    #[value(name = "gpt2-seq-2")]
    GPT2Seq2,
    #[value(name = "gpt2-seq-32")]
    GPT2Seq32,
    #[value(name = "layernorm-32x768")]
    LayerNorm,
    #[value(name = "gelu-32x3072")]
    Gelu,
    #[value(name = "softmax-32x32")]
    Softmax,
    #[value(name = "physics-d8-t10")]
    PhysicsSmall,
    #[value(name = "physics-d16-t10")]
    PhysicsLarge,
    #[value(name = "debug")]
    Debug,
}

#[derive(Debug, Clone, Copy)]
pub struct InstanceSize {
    pub num_inputs: usize,
    pub target_num_cons: usize,
}

// Extremely generous sizing based on number of constraints from Spain's arithmetization
// Does not account for any extra floating point constraints/overhead
pub fn get_instance_size_lower_bound(computation_type: ComputationType) -> InstanceSize {
    let (num_inputs, target_num_cons): (usize, usize) = match computation_type {
        ComputationType::GPT2Seq2 => (10, 18_504_774_754),
        ComputationType::GPT2Seq32 => (10, 22_590_626_044),
        ComputationType::LayerNorm => (49_153, 10_148_320),
        ComputationType::Gelu => (196_609, 47_087_616),
        ComputationType::Softmax => (2_049, 1_307_360),
        ComputationType::PhysicsSmall => (576, 23_243_880),
        ComputationType::PhysicsLarge => (2304, 126_314_280),
        ComputationType::Debug => (10, 259_923),
    };

    InstanceSize {
        num_inputs,
        target_num_cons,
    }
}
