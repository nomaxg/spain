pub mod builder;
pub mod executor;
pub mod fluid;
pub mod r1cs;

#[cfg(test)]
mod tests {
    use crate::executor::PhysicsExampleExecutor;
    use spain::simulate::stateful_simulate;

    #[test]
    fn test_stateful_simulate() {
        let exec = PhysicsExampleExecutor::default();
        stateful_simulate(exec, None);
    }
}
