use crate::broker::Broker;
use anyhow::{Result, anyhow};

// Trait representing a protocol state machine.
// We assume that every message handled triggers a state machine update with a follow-up message.
// "None" indicates that the protocol is over.
pub trait ProtocolState<M> {
    // Handle message, return message for counter-party
    fn handle_message(&mut self, m: &M) -> Option<M>;
    // Init message, should error if the wrong party goes first
    fn init_message(&mut self) -> Result<M>;
}

// Simulates prover + verifier state machine interaction where prover goes first,
pub fn simulate<M, P, V>(mut prover: P, mut verifier: V) -> Result<()>
where
    P: ProtocolState<M>,
    V: ProtocolState<M>,
    M: std::fmt::Debug,
{
    enum Turn {
        Prover,
        Verifier,
    }

    let (mut msg, mut turn) = match prover.init_message() {
        Ok(msg) => (msg, Turn::Verifier),
        Err(prover_err) => match verifier.init_message() {
            Ok(msg) => (msg, Turn::Prover),
            Err(verifier_err) => {
                return Err(anyhow!(
                    "neither party could initiate the protocol: prover={prover_err:?}, verifier={verifier_err:?}"
                ));
            }
        },
    };

    loop {
        dbg!(&msg);
        match turn {
            Turn::Verifier => match verifier.handle_message(&msg) {
                Some(v_msg) => {
                    msg = v_msg;
                    turn = Turn::Prover;
                }
                None => {
                    dbg!("Verifier done.");
                    return Ok(());
                }
            },
            Turn::Prover => match prover.handle_message(&msg) {
                Some(p_msg) => {
                    msg = p_msg;
                    turn = Turn::Verifier;
                }
                None => {
                    dbg!("Prover done.");
                    return Ok(());
                }
            },
        }
    }
}

// Runs a single protocol actor against a broker. The actor can be configured to
// initiate the protocol or wait for the first message.
pub fn run_actor<M, A, B>(actor: &mut A, mut broker: B) -> Result<()>
where
    A: ProtocolState<M>,
    B: Broker<M>,
    M: std::fmt::Debug,
{
    if let Ok(msg) = actor.init_message() {
        broker.send_msg(&msg)?;
    }

    loop {
        let received = match broker.receive_msg() {
            Ok(msg) => msg,
            Err(_) => {
                return Ok(());
            }
        };
        let Some(response) = actor.handle_message(&received) else {
            return Ok(());
        };
        broker.send_msg(&response)?;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::broker::Broker;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;
    // Simple test protocol:
    // 1) Prover pings
    // 2) Verifier pongs
    // 3) Prover acks
    #[derive(Debug, Clone)]
    enum Message {
        Ping,
        Pong,
        Ack,
    }
    struct PingProver;
    struct PingVerifier;
    struct VerifierStarter;
    struct ProverResponder;

    impl ProtocolState<Message> for PingProver {
        fn handle_message(&mut self, m: &Message) -> Option<Message> {
            match m {
                Message::Pong => Some(Message::Ack),
                _ => panic!("prover received unexpected message {:?}", m),
            }
        }
        fn init_message(&mut self) -> Result<Message> {
            Ok(Message::Ping)
        }
    }

    impl ProtocolState<Message> for PingVerifier {
        fn handle_message(&mut self, m: &Message) -> Option<Message> {
            match m {
                Message::Ping => Some(Message::Pong),
                Message::Ack => None,
                _ => None,
            }
        }
        fn init_message(&mut self) -> Result<Message> {
            Err(anyhow!("verifier shouldn't go first"))
        }
    }

    #[test]
    fn simulate_ping() {
        simulate(PingProver, PingVerifier).expect("simulate failed");
    }

    impl ProtocolState<Message> for VerifierStarter {
        fn handle_message(&mut self, m: &Message) -> Option<Message> {
            match m {
                Message::Ack => None,
                _ => panic!("verifier received unexpected message {:?}", m),
            }
        }

        fn init_message(&mut self) -> Result<Message> {
            Ok(Message::Ping)
        }
    }

    impl ProtocolState<Message> for ProverResponder {
        fn handle_message(&mut self, m: &Message) -> Option<Message> {
            match m {
                Message::Ping => Some(Message::Ack),
                _ => panic!("prover received unexpected message {:?}", m),
            }
        }

        fn init_message(&mut self) -> Result<Message> {
            Err(anyhow!("prover shouldn't start in this test"))
        }
    }

    #[test]
    fn simulate_verifier_initiates() {
        simulate(ProverResponder, VerifierStarter)
            .expect("simulate failed when verifier initiated");
    }

    struct MockBroker<M> {
        incoming: VecDeque<M>,
        sent: Rc<RefCell<Vec<M>>>,
    }

    impl<M> MockBroker<M> {
        fn new(incoming: Vec<M>) -> (Self, Rc<RefCell<Vec<M>>>) {
            let sent = Rc::new(RefCell::new(Vec::new()));
            (
                Self {
                    incoming: incoming.into(),
                    sent: sent.clone(),
                },
                sent,
            )
        }
    }

    impl<M: Clone> Broker<M> for MockBroker<M> {
        fn send_msg(&mut self, msg: &M) -> Result<()> {
            self.sent.borrow_mut().push(msg.clone());
            Ok(())
        }

        fn receive_msg(&mut self) -> Result<M> {
            self.incoming
                .pop_front()
                .ok_or_else(|| anyhow!("no more incoming messages"))
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ActorMessage {
        Init,
        Continue,
        Stop,
    }

    struct InitiatingActor;

    impl ProtocolState<ActorMessage> for InitiatingActor {
        fn handle_message(&mut self, m: &ActorMessage) -> Option<ActorMessage> {
            match m {
                ActorMessage::Continue => Some(ActorMessage::Stop),
                ActorMessage::Stop => None,
                ActorMessage::Init => panic!("unexpected init message"),
            }
        }

        fn init_message(&mut self) -> Result<ActorMessage> {
            Ok(ActorMessage::Init)
        }
    }

    #[test]
    fn run_actor_response_correct() {
        let (broker, sent) = MockBroker::new(vec![ActorMessage::Continue, ActorMessage::Stop]);
        let mut actor = InitiatingActor;

        run_actor(&mut actor, broker).expect("run_actor failed");

        assert_eq!(*sent.borrow(), vec![ActorMessage::Init, ActorMessage::Stop]);
    }
}
