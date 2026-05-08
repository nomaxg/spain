use anyhow::{Result, anyhow};
use clap::Parser;
use protocol::broker::JsonBroker;
use protocol::machine::{ProtocolState, run_actor};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct TestMessage(String);

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Test stdout/stdin broker",
    long_about = None
)]
struct Cli {
    // Indicates whether this binary should send the first message
    #[arg(long, default_value_t = false)]
    initiator: bool,
}

struct TestActor {
    initiator: bool,
    counter: usize,
}

impl ProtocolState<TestMessage> for TestActor {
    fn handle_message(&mut self, m: &TestMessage) -> Option<TestMessage> {
        if m.0 == "exit" {
            eprintln!("Terminating.");
            return None;
        }

        if self.initiator {
            self.counter = self.counter.saturating_sub(1);
            eprintln!("counter: {:?}", self.counter);
        }

        let response = if self.initiator && self.counter == 0 {
            TestMessage("exit".to_string())
        } else {
            TestMessage("response!".to_string())
        };
        eprintln!("recv: {:?}", m);
        Some(response)
    }

    fn init_message(&mut self) -> Result<TestMessage> {
        if self.initiator {
            let msg = TestMessage("First message".to_string());
            eprintln!("Initiating with message: {:?}", msg);
            Ok(msg)
        } else {
            Err(anyhow!("non-initiator should wait for first message"))
        }
    }
}

// Responds to quoted "" messages and terminates on "exit"
// Use ncat to test:
// ncat -l 9000 --keep-open -c "../target/release/protocol"
// ncat 127.0.0.1 9000 -c "../target/release/protocol --initiator"
pub fn main() {
    let Cli { initiator } = Cli::parse();
    let mut actor = TestActor {
        initiator,
        counter: 1000,
    };
    let broker = JsonBroker::new();
    run_actor(&mut actor, broker).expect("actor loop failed");
}
