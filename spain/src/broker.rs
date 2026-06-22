use anyhow::Result;
use protocol::broker::{Broker, JsonBroker};
use std::str::FromStr;
use std::time::Duration;

use crate::actor::SpainMessage;
use crate::verifier::ZRangeOpening;

pub struct SpainBroker {
    inner: JsonBroker,
}

impl SpainBroker {
    pub fn new() -> Self {
        Self {
            inner: JsonBroker::new(),
        }
    }
}

impl Default for SpainBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Broker<SpainMessage<T>> for SpainBroker
where
    T: ToString + Clone + FromStr,
    <T as FromStr>::Err: std::fmt::Debug,
{
    fn send_msg(&mut self, msg: &SpainMessage<T>) -> Result<(usize, Duration)> {
        // Public variables should not be included in the proof size accounting
        let discounted_size = match msg {
            SpainMessage::WitnessOpenings(openings) => {
                let discounted_openings: Vec<ZRangeOpening<T>> = openings
                    .iter()
                    .filter(|opening| opening.range_index != 0)
                    .cloned()
                    .collect();
                let discounted_msg = SpainMessage::WitnessOpenings(discounted_openings);
                Some(serde_json::to_vec(&discounted_msg)?.len())
            }
            _ => None,
        };
        let (_, serialization_time) = self.inner.send_msg(msg)?;
        let msg_size = match discounted_size {
            Some(size) => size,
            None => serde_json::to_vec(msg)?.len(),
        };
        Ok((msg_size, serialization_time))
    }

    fn receive_msg(&mut self) -> Result<(SpainMessage<T>, Duration)> {
        self.inner.receive_msg()
    }
}
