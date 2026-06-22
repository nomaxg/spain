pub mod builder;
pub mod executor;
pub mod fluid;
pub mod r1cs;
pub mod utils;
pub mod zklp;

pub use zklp::circuit_zklp;

#[cfg(test)]
mod tests {
    use crate::executor::PhysicsExampleExecutor;
    use spain::simulate::stateful_simulate;

    #[test]
    fn test_stateful_simulate() {
        let exec = PhysicsExampleExecutor::default();
        stateful_simulate::<model::AFloat, _, i128>(exec, None);
    }
}
